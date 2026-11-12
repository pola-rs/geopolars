//! Moving a geometry from one coordinate reference system to another.

use std::cell::OnceCell;
use std::rc::Rc;
use std::sync::Arc;

use geopolars_compute::proj::Transformation;
use geopolars_compute::GeoComputeError;
use geopolars_core::reproject;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::cache::{SharedCache, ThreadCache};
use crate::geoarrow::{describe, Geo};
use crate::proj::specifies_z;

#[derive(Deserialize)]
struct ToCrsKwargs {
    /// The CRS to reproject to, in any form PROJ accepts.
    to: String,
    allow_projection_with_unscaled_z: bool,
    assume_z_in_target_crs: bool,
}

/// The geometry we were handed, with its metadata naming the target CRS.
/// Type operation only.
fn reprojected_geometry_type(dtype: &DataType, to: &str) -> PolarsResult<Geo> {
    let geo = describe(dtype)?;
    let metadata = Arc::unwrap_or_clone(geo.metadata.clone()).with_crs(to);
    Ok(Geo::new(geo.kind, geo.dim, Arc::new(metadata)))
}

/// For [`to_crs`]: the same geometry, but the
/// metadata now names the target CRS, so the dtype is a different one.
fn kernel_output_type_to_crs(input_fields: &[Field], kwargs: ToCrsKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    // Built here as well as in the kernel, so a CRS PROJ cannot use fails while the plan is built.
    let cached = transformation(field.dtype(), &kwargs.to)?;
    reprojects_z(field.dtype(), &kwargs, &cached)?;
    let geo = reprojected_geometry_type(field.dtype(), &kwargs.to)?;
    Ok(Field::new(field.name().clone(), geo.dtype()))
}

/// One transformation per pair of CRSs for the whole process, which every thread clones.
/// Creating one costs milliseconds, more than reprojecting a streaming batch,
/// and threads creating them queue for the one connection PROJ shares to its database.
/// A clone costs a few percent of that.
/// It is cloned under the cache's lock, as only one thread at a time may use a template's PROJ objects.
static TEMPLATES: SharedCache<(String, String), Transformation> = SharedCache::new();

thread_local! {
    /// This thread's clones of [`TEMPLATES`],
    /// as a [`Transformation`] can't be shared between threads.
    static TRANSFORMATIONS: ThreadCache<(String, String), CachedTransformation> =
        ThreadCache::default();
}

/// A transformation, with whether its CRSs specify z, once a geometry with a z needed to know.
/// Looking that up creates a PROJ context and reads PROJ's database, so it is kept too.
struct CachedTransformation {
    transformation: Transformation,
    specify_z: OnceCell<(bool, bool)>,
}

impl CachedTransformation {
    /// Whether `from`, and `to`, each specify z (see [`specifies_z`]).
    /// Not known up front, as [`specifies_z`] fails on some CRSs that reproject fine without z.
    fn specify_z(&self, from: &str, to: &str) -> PolarsResult<(bool, bool)> {
        if let Some(&specify_z) = self.specify_z.get() {
            return Ok(specify_z);
        }
        // One that fails is not cached, and fails again the next time.
        let specify_z = (specifies_z(from)?, specifies_z(to)?);
        Ok(*self.specify_z.get_or_init(|| specify_z))
    }
}

/// The PROJ transformation from the CRS of `dtype` to `to`, cached per thread (see [`TRANSFORMATIONS`]).
fn transformation(dtype: &DataType, to: &str) -> PolarsResult<Rc<CachedTransformation>> {
    let key = (describe(dtype)?.metadata.crs()?.to_owned(), to.to_owned());
    TRANSFORMATIONS.with(|cache| {
        cache.get_or_build(&key, || {
            Ok(CachedTransformation {
                transformation: transformation_from_template(&key)?,
                specify_z: OnceCell::new(),
            })
        })
    })
}

fn transformation_from_template(key: &(String, String)) -> PolarsResult<Transformation> {
    let err = |e: GeoComputeError| polars_err!(ComputeError: "{e}");
    TEMPLATES.with_or_build(
        key,
        || Transformation::new(&key.0, &key.1).map_err(err),
        |template| template.try_clone().map_err(err),
    )
}

/// Whether the `z` of `dtype` goes through PROJ, as in the table of `to_crs` in `geo/crs.py`.
/// Errors where that table needs a flag that is not set.
fn reprojects_z(
    dtype: &DataType,
    kwargs: &ToCrsKwargs,
    cached: &CachedTransformation,
) -> PolarsResult<bool> {
    let geo = describe(dtype)?;
    if !geo.dim.has_z() {
        return Ok(false);
    }
    let (from, to) = (geo.metadata.crs()?, kwargs.to.as_str());
    match cached.specify_z(from, to)? {
        (true, true) => Ok(true),
        (true, false) => {
            polars_ensure!(
                kwargs.allow_projection_with_unscaled_z,
                ComputeError: "{to} does not specify z, so the z of {from} cannot be reprojected to it; \
                pass `allow_projection_with_unscaled_z=True` to keep z as it is"
            );
            Ok(false)
        }
        (false, false) => {
            polars_ensure!(
                kwargs.allow_projection_with_unscaled_z,
                ComputeError: "neither {from} nor {to} specifies z, so z cannot be reprojected; \
                pass `allow_projection_with_unscaled_z=True` to keep z as it is"
            );
            Ok(false)
        }
        (false, true) => {
            polars_ensure!(
                kwargs.assume_z_in_target_crs,
                ComputeError: "{from} does not specify z, so it cannot be reprojected to the z of {to}; \
                pass `assume_z_in_target_crs=True` if z is already in {to}"
            );
            Ok(false)
        }
    }
}

/// See `to_crs`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_to_crs)]
fn to_crs(inputs: &[Series], kwargs: ToCrsKwargs) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let cached = transformation(inputs[0].dtype(), &kwargs.to)?;
    let z = reprojects_z(inputs[0].dtype(), &kwargs, &cached)?;

    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| reproject(chunk.as_ref(), geo.kind, &cached.transformation, z))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is still the storage.
    let out = unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    };

    let typ = reprojected_geometry_type(inputs[0].dtype(), &kwargs.to)?.instance();
    Ok(out.into_extension(typ))
}
