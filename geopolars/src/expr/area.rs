//! The area a geometry encloses.
//! Implements the Python `area`.

use std::fmt::Display;

use geopolars_compute::native::AreaMetric;
use geopolars_compute::Planar;
use geopolars_core::{areas, PolygonKind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::distance::{geodesic_metric_of, DistanceMethod};
use crate::geoarrow::describe;

/// The keyword arguments of the Python `area`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AreaKwargs {
    distance_method: DistanceMethod,
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

/// The area of every geometry in `geometries`, as `metric` measures it.
fn measure(metric: &impl AreaMetric<Error: Display>, geometries: &Series) -> PolarsResult<Series> {
    let kind = PolygonKind::new("area", describe(geometries.dtype())?.kind)?;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| areas(chunk.as_ref(), kind, metric));
    Ok(Float64Chunked::try_from_chunk_iter(geometries.name().clone(), chunks)?.into_series())
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_area)]
fn area(inputs: &[Series], kwargs: AreaKwargs) -> PolarsResult<Series> {
    match kwargs.distance_method {
        DistanceMethod::Planar => measure(&Planar, &inputs[0]),
        DistanceMethod::Geodesic => {
            let metric = geodesic_metric_of(&inputs[0].field())?;
            measure(&*metric, &inputs[0])
        }
    }
}
