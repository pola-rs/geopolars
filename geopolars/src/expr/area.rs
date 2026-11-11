//! The area a geometry encloses:
//! - on the plane its coordinates lie in ([`area_planar`]),
//! - along the ellipsoid of its CRS ([`area_geodesic`]).

use std::fmt::Display;

use geopolars_compute::native::AreaMetric;
use geopolars_compute::Planar;
use geopolars_core::{Kind, MultiPolygonArray, PolygonArray};
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

/// Only polygons and multipolygons enclose anything: other kinds are refused
/// rather than given an area of `0.0`.
fn encloses_nothing(kind: Kind) -> PolarsError {
    polars_err!(SchemaMismatch: "area does not accept a `{}` column", kind.name())
}

/// An `f64` per geometry: for a multipolygon, the sum of its parts.
fn area_dtype(field: &Field) -> PolarsResult<Field> {
    match describe(field.dtype())?.kind {
        Kind::Polygon | Kind::MultiPolygon => {
            Ok(Field::new(field.name().clone(), DataType::Float64))
        }
        kind @ (Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString) => {
            Err(encloses_nothing(kind))
        }
    }
}

/// For [`area_planar`]: in the unit of the coordinates squared.
fn kernel_output_type_area_planar(input_fields: &[Field]) -> PolarsResult<Field> {
    area_dtype(&input_fields[0])
}

/// For [`area_geodesic`]: in the unit of the CRS squared.
fn kernel_output_type_area_geodesic(input_fields: &[Field]) -> PolarsResult<Field> {
    // Check if the CRS can be used by PROJ. If not, raise error at plan time rather than at runtime.
    geodesic_metric(&crs_of(&input_fields[0])?)?;
    area_dtype(&input_fields[0])
}

/// The area of every geometry in `geometries`, as `metric` measures it.
fn measure(metric: &impl AreaMetric<Error: Display>, geometries: &Series) -> PolarsResult<Series> {
    let multi = match describe(geometries.dtype())?.kind {
        Kind::Polygon => false,
        Kind::MultiPolygon => true,
        kind @ (Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString) => {
            return Err(encloses_nothing(kind))
        }
    };
    let storage = geometries.ext()?.storage();
    let chunks = storage.list()?.downcast_iter().map(|rows| {
        let areas = if multi {
            let array = MultiPolygonArray::new_untrimmed(rows)?;
            geopolars_compute::native::multi_polygon_areas(metric, &array)
        } else {
            let array = PolygonArray::new_untrimmed(rows)?;
            geopolars_compute::native::polygon_areas(metric, &array)
        }
        .map_err(|e| polars_err!(ComputeError: "{e}"))?;
        PolarsResult::Ok(areas)
    });
    Ok(Float64Chunked::try_from_chunk_iter(geometries.name().clone(), chunks)?.into_series())
}

/// See `area_planar`.
#[polars_expr(output_type_func=kernel_output_type_area_planar)]
fn area_planar(inputs: &[Series]) -> PolarsResult<Series> {
    measure(&Planar, &inputs[0])
}

/// See `area`.
#[polars_expr(output_type_func=kernel_output_type_area_geodesic)]
fn area_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let metric = geodesic_metric(&crs_of(&inputs[0].field())?)?;
    measure(&*metric, &inputs[0])
}
