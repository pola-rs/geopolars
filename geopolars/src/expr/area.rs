//! The area a geometry encloses.
//! Implements the Python `area_planar` and `area_geodesic`.

use std::fmt::Display;

use geopolars_compute::native::AreaMetric;
use geopolars_compute::Planar;
use geopolars_core::{areas, PolygonKind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use super::distance::geodesic_metric;
use crate::geoarrow::describe;

/// The CRS of a column to measure an area in. Type operation only.
fn crs_of(field: &Field) -> PolarsResult<String> {
    let geo = describe(field.dtype())?;
    polars_ensure!(
        geo.metadata.declares_crs(),
        SchemaMismatch: "the geometry declares no CRS, so there is no ellipsoid to measure on"
    );
    geo.metadata.crs().map(str::to_owned)
}

/// Refuses every kind but polygons and multipolygons,
/// as the Python side gives the others their `0.0` itself.
fn area_dtype(field: &Field) -> PolarsResult<Field> {
    PolygonKind::new("area", describe(field.dtype())?.kind)?;
    Ok(Field::new(field.name().clone(), DataType::Float64))
}

fn kernel_output_type_area_planar(input_fields: &[Field]) -> PolarsResult<Field> {
    area_dtype(&input_fields[0])
}

fn kernel_output_type_area_geodesic(input_fields: &[Field]) -> PolarsResult<Field> {
    // Check if the CRS can be used by PROJ. If not, raise error at plan time rather than at runtime.
    geodesic_metric(&crs_of(&input_fields[0])?)?;
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

#[polars_expr(output_type_func=kernel_output_type_area_planar)]
fn area_planar(inputs: &[Series]) -> PolarsResult<Series> {
    measure(&Planar, &inputs[0])
}

#[polars_expr(output_type_func=kernel_output_type_area_geodesic)]
fn area_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let metric = geodesic_metric(&crs_of(&inputs[0].field())?)?;
    measure(&*metric, &inputs[0])
}
