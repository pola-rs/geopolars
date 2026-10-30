//! Snapping every coordinate onto a grid, through `geopolars_compute::native::*_snapped_to_grid`.
//! See `snap_to_grid` in Python for the rules.

use geopolars_compute::native::{
    line_strings_snapped_to_grid, multi_line_strings_snapped_to_grid, multi_points_snapped_to_grid,
    multi_polygons_snapped_to_grid, points_snapped_to_grid, polygons_snapped_to_grid, Grid,
    RoundMode,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::{describe, Kind};

#[derive(Deserialize)]
#[serde(remote = "RoundMode", rename_all = "snake_case")]
enum RoundModeDef {
    HalfToEven,
    HalfAwayFromZero,
    ToZero,
}

/// See [`Grid`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapToGridKwargs {
    spacing: [f64; 3],
    origin: [f64; 3],
    #[serde(with = "RoundModeDef")]
    mode: RoundMode,
}

fn chunk_snapped_to_grid(
    chunk: &dyn Array,
    kind: Kind,
    grid: &Grid,
) -> PolarsResult<Box<dyn Array>> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_snapped_to_grid(&PointArray::new(Coords::new(chunk)?), grid)
            .into_struct_array()
            .boxed(),
        Kind::LineString => line_strings_snapped_to_grid(&LineStringArray::new(rows()?)?, grid)
            .into_list_array()
            .boxed(),
        Kind::MultiPoint => multi_points_snapped_to_grid(&MultiPointArray::new(rows()?)?, grid)
            .into_list_array()
            .boxed(),
        Kind::Polygon => polygons_snapped_to_grid(&PolygonArray::new(rows()?)?, grid)
            .into_list_array()
            .boxed(),
        Kind::MultiLineString => {
            multi_line_strings_snapped_to_grid(&MultiLineStringArray::new(rows()?)?, grid)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            multi_polygons_snapped_to_grid(&MultiPolygonArray::new(rows()?)?, grid)
                .into_list_array()
                .boxed()
        }
    })
}

/// See `snap_to_grid`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn snap_to_grid(inputs: &[Series], kwargs: SnapToGridKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let SnapToGridKwargs {
        spacing,
        origin,
        mode,
    } = kwargs;
    let grid = Grid {
        spacing,
        origin,
        mode,
    };
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_snapped_to_grid(chunk.as_ref(), kind, &grid))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
