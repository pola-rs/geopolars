//! Whether two geometries are equal, coordinate by coordinate
//! (see [`geopolars_compute::native::points_are_equal`] and the kernels of the other kinds).

use geopolars_compute::native::{
    line_strings_are_equal, multi_line_strings_are_equal, multi_points_are_equal,
    multi_polygons_are_equal, points_are_equal, polygons_are_equal, Coincidence,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_core::utils::align_chunks_binary;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, ensure_same_crs, Kind};

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsEqualKwargs {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    nans_equal: bool,
    unique_z: bool,
    unique_m: bool,
}

/// For [`is_equal`]: a `bool`, named after the first geometry.
///
/// Both have to be the same kind and dimension, in the same CRS (or neither declaring one).
fn kernel_output_type_is_equal(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "is_equal takes two geometries, got {} inputs", input_fields.len());
    };
    let (geo_a, geo_b) = (describe(a.dtype())?, describe(b.dtype())?);
    polars_ensure!(
        geo_a.kind == geo_b.kind && geo_a.dim == geo_b.dim,
        SchemaMismatch: "is_equal expects two geometries of the same type, got: {} and {}",
        a.dtype(), b.dtype()
    );
    ensure_same_crs(&geo_a, &geo_b)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

/// The kernel of `kind` on one chunk of `a`'s storage and one of `b`'s.
/// `b` is as long as `a`, or a single row.
fn chunk_is_equal(
    a: &dyn Array,
    b: &dyn Array,
    kind: Kind,
    kwargs: &IsEqualKwargs,
) -> PolarsResult<BooleanArray> {
    let coincidence = Coincidence::new(kwargs.abs_tol, kwargs.rel_tol);
    let (unique_z, unique_m, nans_equal) = (kwargs.unique_z, kwargs.unique_m, kwargs.nans_equal);
    let rows = |chunk| downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_are_equal(
            &PointArray::new(Coords::new(a)?),
            &PointArray::new(Coords::new(b)?),
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiPoint => multi_points_are_equal(
            &MultiPointArray::new_untrimmed(rows(a)?)?,
            &MultiPointArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::LineString => line_strings_are_equal(
            &LineStringArray::new_untrimmed(rows(a)?)?,
            &LineStringArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiLineString => multi_line_strings_are_equal(
            &MultiLineStringArray::new_untrimmed(rows(a)?)?,
            &MultiLineStringArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::Polygon => polygons_are_equal(
            &PolygonArray::new_untrimmed(rows(a)?)?,
            &PolygonArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiPolygon => multi_polygons_are_equal(
            &MultiPolygonArray::new_untrimmed(rows(a)?)?,
            &MultiPolygonArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
    })
}

/// The chunks of the storages of `a` and `b`, which are the same length, split at the same rows.
fn aligned_chunks(a: &Series, b: &Series) -> PolarsResult<Vec<(ArrayRef, ArrayRef)>> {
    let (a, b) = (a.ext()?.storage(), b.ext()?.storage());
    let zipped =
        |a: &[ArrayRef], b: &[ArrayRef]| a.iter().cloned().zip(b.iter().cloned()).collect();
    Ok(match a.dtype() {
        DataType::Struct(_) => {
            let (a, b) = align_chunks_binary(a.struct_()?, b.struct_()?);
            zipped(a.chunks(), b.chunks())
        }
        _ => {
            let (a, b) = align_chunks_binary(a.list()?, b.list()?);
            zipped(a.chunks(), b.chunks())
        }
    })
}

/// [chunk_is_equal] for every chunk of `many`, against the single row of `one`.
/// Equality goes both ways, so this serves a single geometry on either side.
fn every_against_one(
    one: &Series,
    many: &Series,
    kind: Kind,
    kwargs: &IsEqualKwargs,
) -> PolarsResult<Vec<BooleanArray>> {
    let one = one.ext()?.storage().rechunk();
    let one = one.chunks()[0].as_ref();
    many.ext()?
        .storage()
        .chunks()
        .iter()
        .map(|many| chunk_is_equal(many.as_ref(), one, kind, kwargs))
        .collect()
}

/// See `is_equal`.
#[polars_expr(output_type_func=kernel_output_type_is_equal)]
fn is_equal(inputs: &[Series], kwargs: IsEqualKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let kind = describe(a.dtype())?.kind;
    let chunks = match (a.len(), b.len()) {
        (n, m) if n == m => aligned_chunks(a, b)?
            .iter()
            .map(|(a, b)| chunk_is_equal(a.as_ref(), b.as_ref(), kind, &kwargs))
            .collect::<PolarsResult<Vec<_>>>()?,
        (1, _) => every_against_one(a, b, kind, &kwargs)?,
        (_, 1) => every_against_one(b, a, kind, &kwargs)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot compare {n} geometries against {m}"),
    };
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
