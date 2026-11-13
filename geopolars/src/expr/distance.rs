//! How far apart two points are.
//! Implements the Python `distance`.
//! Also holds the [`geodesic_metric`] cache, which `area` and `length` measure with too.

use std::fmt::Display;
use std::rc::Rc;

use geopolars_compute::native::DistanceMetric;
use geopolars_compute::{GeodesicMetric, Planar};
use geopolars_core::{distances_to_one, pairwise_distances};
use geopolars_views::Kind;
use polars::prelude::*;
use polars_arrow::array::PrimitiveArray;
use polars_arrow::datatypes::ArrowDataType;
use proj::Proj;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::cache::ThreadCache;
use crate::geoarrow::{aligned_pairs, describe, ensure_same_crs_unless_bare, GeometryPairs};
use crate::proj::GeodeticCrs;

thread_local! {
    /// Per thread, because a [`Proj`] can't be shared between threads.
    static METRICS: ThreadCache<String, GeodesicMetric> = ThreadCache::default();
}

/// Populates or gets the metric for `crs` from the cache.
///
/// `crs` is anything PROJ accepts, as [`ExtensionMetadata::crs`](crate::geoarrow::crs::ExtensionMetadata::crs) gives it.
pub fn geodesic_metric(crs: &str) -> PolarsResult<Rc<GeodesicMetric>> {
    METRICS.with(|metrics| metrics.get_or_build(crs, || build(crs)))
}

/// The [`geodesic_metric`] for the CRS a geometry `field` declares.
/// A field that declares no CRS is refused, as there is no ellipsoid to measure on.
pub(crate) fn geodesic_metric_of(field: &Field) -> PolarsResult<Rc<GeodesicMetric>> {
    let geo = describe(field.dtype())?;
    polars_ensure!(
        geo.metadata.declares_crs(),
        SchemaMismatch: "the geometry declares no CRS, so there is no ellipsoid to measure on"
    );
    geodesic_metric(geo.metadata.crs()?)
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

/// The `method` of the Python `distance` and `length`, which the Python side always resolves.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum DistanceMethod {
    /// See [`Planar`].
    Planar,
    /// See [`GeodesicMetric`].
    Geodesic,
}

/// The keyword arguments of the Python `distance`.
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

/// Measures the distance between the points of `inputs[0]` and `inputs[1]` with `metric`.
fn measure(
    metric: &impl DistanceMetric<Error: Display>,
    inputs: &[Series],
) -> PolarsResult<Series> {
    let chunks = match aligned_pairs(&inputs[0], &inputs[1])? {
        GeometryPairs::Pairwise(pairs) => pairs
            .iter()
            .map(|(a, b)| pairwise_distances(a.as_ref(), b.as_ref(), metric))
            .collect::<PolarsResult<Vec<_>>>()?,
        // Distance goes both ways, so it does not matter which side `one` came from.
        GeometryPairs::OneAgainstMany { a: one, b: many }
        | GeometryPairs::ManyAgainstOne { a: many, b: one } => many
            .iter()
            .map(|many| distances_to_one(one.as_ref(), many.as_ref(), metric))
            .collect::<PolarsResult<Vec<_>>>()?,
        GeometryPairs::AllNull(lengths) => lengths
            .iter()
            .map(|&rows| PrimitiveArray::new_null(ArrowDataType::Float64, rows))
            .collect(),
    };
    Ok(Float64Chunked::from_chunk_iter(inputs[0].name().clone(), chunks).into_series())
}

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
    use crate::cache::CACHE_CAPACITY;

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
        for zone in 1..=CACHE_CAPACITY + 1 {
            geodesic_metric(&format!("EPSG:{}", 32600 + zone)).unwrap();
        }

        assert!(METRICS.with(ThreadCache::len) <= CACHE_CAPACITY);
    }
}
