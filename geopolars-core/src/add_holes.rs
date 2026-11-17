use geopolars_compute::native::{
    have_room_for_holes, polygons_with_added_holes, polygons_with_empty_exteriors,
};
use geopolars_compute::{EmptyPolygonExterior, GeoComputeError, NullParts, OnRowError};
use geopolars_views::{downcast, GeometryArray, PolygonArray};
use polars_arrow::array::Array;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_core::prelude::PolarsResult;

use crate::parts::{lines_per_row, rows_validity, validated_polygons, CloseRings};
use crate::IntoPolarsResult;

fn null_count(validity: &Option<Bitmap>) -> usize {
    validity.as_ref().map_or(0, Bitmap::unset_bits)
}

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
    let polygons = match empty_polygon_exterior {
        // A polygon without rings would take its first hole as its exterior.
        EmptyPolygonExterior::Keep => {
            let room = have_room_for_holes(&polygons, &holes);
            polygons_with_empty_exteriors(&polygons, &!&room)
        }
        _ => polygons,
    };
    let (holes, validity) = match empty_polygon_exterior {
        EmptyPolygonExterior::Keep => (holes, validity),
        EmptyPolygonExterior::Empty => {
            let room = have_room_for_holes(&polygons, &holes);
            (holes.only_on_rows(&room), validity)
        }
        mode @ (EmptyPolygonExterior::Null | EmptyPolygonExterior::Error) => {
            let room = have_room_for_holes(&polygons, &holes);
            let with_room = combine_validities_and(validity.as_ref(), Some(&room));
            // A row that is null already has no exterior to complain about.
            if mode == EmptyPolygonExterior::Error
                && on_row_error == OnRowError::Raise
                && null_count(&with_room) > null_count(&validity)
            {
                return Err(GeoComputeError::HoleInEmptyExterior).into_polars();
            }
            (holes, with_room)
        }
    };
    let added = polygons_with_added_holes(&polygons, Some(&holes), validity);
    Ok(GeometryArray::Polygon(validated_polygons(
        added,
        close,
        empty_polygon_exterior,
    )))
}
