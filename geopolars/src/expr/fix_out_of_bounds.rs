//! Fixing the coordinates outside the CRS's bounds,
//! through `geopolars_compute::native::*_with_selected`, or `*_with_clamped_coords` to clamp.
//! See `fix_out_of_bounds` in Python for the rules.

use geopolars_compute::native::{
    line_strings_with_clamped_coords, multi_line_strings_with_clamped_coords,
    multi_points_with_clamped_coords, multi_polygons_with_clamped_coords,
    points_with_clamped_coords, polygons_with_clamped_coords, Bounds, CoordinateSelection,
    OutOfBoundsAction, TooShort,
};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use super::filter_coordinates::chunk_with_selected;
use super::remove_repeats_in_succession::TooShortDef;
use crate::geoarrow::describe;

#[derive(Deserialize)]
#[serde(remote = "OutOfBoundsAction", rename_all = "snake_case")]
enum OutOfBoundsActionDef {
    SetEmpty,
    Drop,
    SetNull,
    Clamp,
}

/// See `OutOfBounds` in python, with its `crs_bounds` looked up as `bounds`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixOutOfBoundsKwargs {
    bounds: Bounds,
    #[serde(with = "OutOfBoundsActionDef")]
    action: OutOfBoundsAction,
    #[serde(with = "TooShortDef")]
    too_short: TooShort,
}

/// [`fix_out_of_bounds`] with [OutOfBoundsAction::Clamp], for one chunk of the storage of a `kind`.
fn chunk_with_clamped_coords(
    chunk: &dyn Array,
    kind: Kind,
    bounds: &Bounds,
) -> PolarsResult<Box<dyn Array>> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_with_clamped_coords(&PointArray::new(Coords::new(chunk)?), bounds)
            .into_struct_array()
            .boxed(),
        Kind::LineString => {
            line_strings_with_clamped_coords(&LineStringArray::new(rows()?)?, bounds)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPoint => {
            multi_points_with_clamped_coords(&MultiPointArray::new(rows()?)?, bounds)
                .into_list_array()
                .boxed()
        }
        Kind::Polygon => polygons_with_clamped_coords(&PolygonArray::new(rows()?)?, bounds)
            .into_list_array()
            .boxed(),
        Kind::MultiLineString => {
            multi_line_strings_with_clamped_coords(&MultiLineStringArray::new(rows()?)?, bounds)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            multi_polygons_with_clamped_coords(&MultiPolygonArray::new(rows()?)?, bounds)
                .into_list_array()
                .boxed()
        }
    })
}

/// See `fix_out_of_bounds`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn fix_out_of_bounds(inputs: &[Series], kwargs: FixOutOfBoundsKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let FixOutOfBoundsKwargs {
        bounds,
        action,
        too_short,
    } = kwargs;
    let selection = CoordinateSelection {
        non_finite: None,
        bounds: Some(bounds),
    };
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| match action.selected_action() {
            Some(action) => {
                chunk_with_selected(chunk.as_ref(), kind, &selection, action, too_short)
            }
            None => chunk_with_clamped_coords(chunk.as_ref(), kind, &bounds),
        })
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
