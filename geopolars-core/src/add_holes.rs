use geopolars_compute::native::holes_added_to_polygons;
use geopolars_compute::{EmptyPolygonExterior, NullParts, OnRowError};
use geopolars_views::{downcast, GeometryArray, PolygonArray};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::parts::{lines_per_row, rows_validity, validated_polygons, CloseRings};
use crate::IntoPolarsResult;

/// Adds `holes` to `polygons`, one chunk of each.
/// Implements the Python `add_holes`.
/// `holes` only holds a single ring per row if `listed=false`.
pub fn add_holes(
    polygons: &dyn Array,
    holes: &dyn Array,
    listed: bool,
    null_parts: NullParts,
    empty_polygon_exterior: EmptyPolygonExterior,
    on_row_error: OnRowError,
    close: Option<CloseRings>,
) -> PolarsResult<GeometryArray> {
    let polygons = PolygonArray::new(downcast(polygons, "a list of rings")?)?;
    let holes = lines_per_row(holes, listed, null_parts)?;
    let validity = rows_validity(&polygons.rows, holes.rows(), null_parts);
    let added = holes_added_to_polygons(
        &polygons,
        holes,
        validity,
        empty_polygon_exterior,
        on_row_error,
    )
    .into_polars()?;
    let holes_in_empty_exterior = empty_polygon_exterior == EmptyPolygonExterior::Keep;
    Ok(GeometryArray::Polygon(validated_polygons(
        added,
        close,
        holes_in_empty_exterior,
    )))
}
