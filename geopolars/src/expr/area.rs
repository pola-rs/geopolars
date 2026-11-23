//! The area a geometry encloses.
//! Implements the Python `area`.

use geopolars_compute::native::{AreaMetric, Bounds, NegativeArea};
use geopolars_compute::{EarlyCoordCheck, GeoComputeError, OnOutOfBounds, OnRowError, Planar};
use geopolars_core::{areas, PolygonKind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::distance::{geodesic_metric_of, DistanceMethod};
use crate::common_arguments::{EarlyCoordCheckDef, OnOutOfBoundsDef, OnRowErrorDef};
use crate::geoarrow::describe;

#[derive(Debug, Deserialize)]
#[serde(remote = "NegativeArea", rename_all = "snake_case")]
enum NegativeAreaDef {
    Zero,
    Keep,
}

/// The keyword arguments of the Python `area`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AreaKwargs {
    distance_method: DistanceMethod,
    #[serde(with = "OnRowErrorDef")]
    on_row_error: OnRowError,
    #[serde(with = "EarlyCoordCheckDef")]
    early_coord_check: EarlyCoordCheck,
    /// The bounds `crs_bounds` names, looked up at plan time.
    bounds: Option<Bounds>,
    #[serde(with = "OnOutOfBoundsDef")]
    on_out_of_bounds: OnOutOfBounds,
    #[serde(with = "NegativeAreaDef")]
    negative_area: NegativeArea,
}

/// Refuses every kind but polygons and multipolygons,
/// as the Python side gives the others their `0.0` itself.
fn area_dtype(field: &Field) -> PolarsResult<Field> {
    PolygonKind::new("area", describe(field.dtype())?.kind)?;
    Ok(Field::new(field.name().clone(), DataType::Float64))
}

fn kernel_output_type_area(input_fields: &[Field], kwargs: AreaKwargs) -> PolarsResult<Field> {
    if let DistanceMethod::Geodesic = kwargs.distance_method {
        // So a CRS PROJ cannot measure in fails while the plan is built.
        geodesic_metric_of(&input_fields[0])?;
    }
    area_dtype(&input_fields[0])
}

fn measure_areas(
    metric: &impl AreaMetric<Error: Into<GeoComputeError>>,
    geometries: &Series,
    kwargs: &AreaKwargs,
) -> PolarsResult<Series> {
    let kind = PolygonKind::new("area", describe(geometries.dtype())?.kind)?;
    let storage = geometries.ext()?.storage();
    let chunks = storage.chunks().iter().map(|chunk| {
        areas(
            chunk.as_ref(),
            kind,
            metric,
            kwargs.on_row_error,
            kwargs.early_coord_check,
            kwargs.bounds,
            kwargs.on_out_of_bounds,
            kwargs.negative_area,
        )
    });
    Ok(Float64Chunked::try_from_chunk_iter(geometries.name().clone(), chunks)?.into_series())
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_area)]
fn area(inputs: &[Series], kwargs: AreaKwargs) -> PolarsResult<Series> {
    match kwargs.distance_method {
        DistanceMethod::Planar => measure_areas(&Planar, &inputs[0], &kwargs),
        DistanceMethod::Geodesic => {
            let metric = geodesic_metric_of(&inputs[0].field())?;
            measure_areas(&*metric, &inputs[0], &kwargs)
        }
    }
}
