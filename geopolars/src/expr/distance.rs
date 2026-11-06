//! How far apart points are.
//! Uses [DistanceMetric] to choose the how to measure distance (Planar/geodesic etc)
//! Wraps functions from [geopolars_compute::native::distance]
//!

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Display;
use std::rc::Rc;

use geopolars_compute::native::DistanceMetric;
use geopolars_compute::with_const_flags;
use geopolars_compute::{GeodesicMetric, Planar};
use geopolars_core::{Coords, PointArray};
use polars::prelude::*;
use polars_arrow::array::PrimitiveArray;
use polars_arrow::datatypes::ArrowDataType;
use proj::Proj;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{aligned_pairs, describe, ensure_same_crs_unless_bare, GeometryPairs, Kind};
use crate::proj::GeodeticCrs;

/// How many CRSes to keep in cache.
/// If more are used, it will start over.
const CACHED_CRSS: usize = 16;

thread_local! {
    // TODO: use pl hashmap, is faster.
    // TODO: make a second 'cache' which literally stores the pointer of the string,
    // so that it can re-use very quickly for the last used CRS (same loc in memory from header)
    /// A cache of CRSes.
    /// call to PROJ by parsing costs more than measuring a small morsel, so cached.
    /// Per thread, because a [`Proj`] can't be shared between threads.
    /// With this, CRS can be re-used over multiple morsels.
    static METRICS: RefCell<HashMap<String, Rc<GeodesicMetric>>> = RefCell::new(HashMap::new());
}

/// Populates or gets the metric for `crs` from the cache.
///
/// `crs` is anything PROJ accepts, as [`ExtensionMetadata::crs`](crate::geoarrow::crs::ExtensionMetadata::crs) gives it.
pub fn geodesic_metric(crs: &str) -> PolarsResult<Rc<GeodesicMetric>> {
    // A CRS that fails to build is not cached, and fails again the next time.
    if let Some(metric) = METRICS.with_borrow(|metrics| metrics.get(crs).cloned()) {
        return Ok(metric);
    }
    // Built outside the borrow, so nothing is held while PROJ runs.
    let metric = Rc::new(build(crs)?);
    METRICS.with_borrow_mut(|metrics| {
        if metrics.len() >= CACHED_CRSS {
            metrics.clear();
        }
        metrics.insert(crs.to_owned(), Rc::clone(&metric));
    });
    Ok(metric)
}

fn build(crs: &str) -> PolarsResult<GeodesicMetric> {
    let geodetic = GeodeticCrs::of(crs)?;
    let to_lonlat = Proj::new_known_crs(crs, &geodetic.definition, None).map_err(
        |e| polars_err!(ComputeError: "cannot measure along the ellipsoid in {crs}: {e}"),
    )?;
    Ok(GeodesicMetric::new(
        to_lonlat,
        geodetic.semi_major,
        geodetic.flattening,
    ))
}

/// How `distance` measures, as passed from Python.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum DistanceMethod {
    /// See [`Planar`].
    Planar,
    /// See [`GeodesicMetric`].
    Geodesic,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DistanceKwargs {
    pub(crate) method: DistanceMethod,
    // TODO: flag which allows automatically re-projecting points if they are in a different CRS.
    // Should be false by default (which raises plan-time error)
}

/// The CRS two point columns share, or `None` if neither declares one. Type operation only.
///
/// Distances between points in different CRSs would have to pick one of them,
/// so they are refused rather than silently reprojected.
fn shared_crs(fields: &[Field]) -> PolarsResult<Option<String>> {
    let [a, b] = fields else {
        polars_bail!(ComputeError: "a distance is between two points, got {} inputs", fields.len());
    };
    let mut metadata = Vec::with_capacity(2);
    for field in [a, b] {
        let geo = describe(field.dtype())?;
        polars_ensure!(
            geo.kind == Kind::Point,
            SchemaMismatch: "a distance is measured between two `{}` columns, got: {}",
            Kind::Point.name(), field.dtype()
        );
        metadata.push(geo.metadata);
    }

    let [a, b] = [&metadata[0], &metadata[1]];
    ensure_same_crs_unless_bare(Some(a), Some(b), ("first point", "second point"))?;
    Ok(a.declares_crs()
        .then(|| a.crs())
        .transpose()?
        .map(str::to_owned))
}

/// The CRS to measure along the ellipsoid of. Type operation only.
fn geodesic_crs(fields: &[Field]) -> PolarsResult<String> {
    shared_crs(fields)?.ok_or_else(|| {
        polars_err!(
            SchemaMismatch: "neither point declares a CRS, so there is no ellipsoid to measure on"
        )
    })
}

/// Distance is returned as `f64`, named after the first point.
fn kernel_output_type_distance(
    input_fields: &[Field],
    kwargs: DistanceKwargs,
) -> PolarsResult<Field> {
    match kwargs.method {
        DistanceMethod::Planar => {
            shared_crs(input_fields)?;
        }
        DistanceMethod::Geodesic => {
            // So a CRS PROJ cannot measure in fails while the plan is built.
            geodesic_metric(&geodesic_crs(input_fields)?)?;
        }
    }
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Float64,
    ))
}

/// [geopolars_compute::native::pairwise], into a polars array.
fn pairwise<const Z: bool>(
    metric: &impl DistanceMetric<Error: Display>,
    a: &PointArray,
    b: &PointArray,
) -> PolarsResult<PrimitiveArray<f64>> {
    geopolars_compute::native::pairwise::<Z, true, _>(metric, a, b)
        .map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// [geopolars_compute::native::against_one], into a polars array.
fn against_one<const Z: bool>(
    metric: &impl DistanceMetric<Error: Display>,
    one: [f64; 3],
    many: &PointArray,
) -> PolarsResult<PrimitiveArray<f64>> {
    geopolars_compute::native::against_one::<Z, true, _>(metric, one, many)
        .map_err(|e| polars_err!(ComputeError: "{e}"))
}

fn points(chunk: &ArrayRef) -> PolarsResult<PointArray> {
    Ok(PointArray::new(Coords::new(chunk.as_ref())?))
}

fn measure_chunks<const Z: bool>(
    metric: &impl DistanceMetric<Error: Display>,
    a: &Series,
    b: &Series,
) -> PolarsResult<Vec<PrimitiveArray<f64>>> {
    Ok(match aligned_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs
            .iter()
            .map(|(a, b)| pairwise::<Z>(metric, &points(a)?, &points(b)?))
            .collect::<PolarsResult<Vec<_>>>()?,
        GeometryPairs::AgainstOne { one, many, .. } => {
            let one = points(&one)?
                .get(0)
                .position::<false, Z>()
                .expect("`NULLS=false` always gives one");
            many.iter()
                .map(|many| against_one::<Z>(metric, one, &points(many)?))
                .collect::<PolarsResult<Vec<_>>>()?
        }
        GeometryPairs::AllNull(lengths) => lengths
            .iter()
            .map(|&rows| PrimitiveArray::new_null(ArrowDataType::Float64, rows))
            .collect(),
    })
}

/// The distance between the points of `inputs[0]` and `inputs[1]`, measured with `metric`.
fn measure(
    metric: &impl DistanceMetric<Error: Display>,
    inputs: &[Series],
) -> PolarsResult<Series> {
    let z = describe(inputs[0].dtype())?.dim.has_z() && describe(inputs[1].dtype())?.dim.has_z();
    let chunks = with_const_flags!(Z = z; measure_chunks::<Z>(metric, &inputs[0], &inputs[1]))?;
    Ok(Float64Chunked::from_chunk_iter(inputs[0].name().clone(), chunks).into_series())
}

/// See `distance`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_distance)]
fn distance(inputs: &[Series], kwargs: DistanceKwargs) -> PolarsResult<Series> {
    let fields = [
        inputs[0].field().into_owned(),
        inputs[1].field().into_owned(),
    ];
    match kwargs.method {
        DistanceMethod::Planar => {
            shared_crs(&fields)?;
            measure(&Planar, inputs)
        }
        DistanceMethod::Geodesic => measure(&*geodesic_metric(&geodesic_crs(&fields)?)?, inputs),
    }
}

#[cfg(test)]
mod tests {
    //! What the Python suite cannot see: whether a metric was built again.

    use super::*;

    #[test]
    fn a_thread_builds_the_metric_for_a_crs_once() {
        let first = geodesic_metric("EPSG:4326").unwrap();

        assert!(Rc::ptr_eq(&first, &geodesic_metric("EPSG:4326").unwrap()));
        assert!(!Rc::ptr_eq(&first, &geodesic_metric("EPSG:4289").unwrap()));
    }

    #[test]
    fn a_crs_that_fails_is_not_cached() {
        assert!(geodesic_metric("EPSG:4978").is_err());
        assert!(geodesic_metric("EPSG:4978").is_err());
    }

    #[test]
    fn the_cache_starts_over_rather_than_growing() {
        for zone in 1..=CACHED_CRSS + 1 {
            geodesic_metric(&format!("EPSG:{}", 32600 + zone)).unwrap();
        }

        assert!(METRICS.with_borrow(|metrics| metrics.len()) <= CACHED_CRSS);
    }
}
