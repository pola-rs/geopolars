use geopolars_compute::native::{have_room_for_holes, polygons_with_added_holes};
use geopolars_views::{downcast, GeometryArray, PolygonArray};
use polars_arrow::array::Array;
use polars_arrow::compute::utils::combine_validities_and;
use polars_core::prelude::PolarsResult;

use crate::parts::{lines_per_row, rows_validity, validated_polygons, CloseRings, NullParts};

/// `empty_exterior` of the Python `add_holes`.
#[derive(Debug, Clone, Copy, Default)]
pub enum EmptyExterior {
    #[default]
    Null,
    Keep,
}

/// Adds `holes` to `polygons`, one chunk of each.
/// Implements the Python `add_holes`.
/// `holes` only holds a single ring per row if `listed=false`.
pub fn add_holes(
    polygons: &dyn Array,
    holes: &dyn Array,
    listed: bool,
    null_parts: NullParts,
    empty_exterior: EmptyExterior,
    close: Option<CloseRings>,
) -> PolarsResult<GeometryArray> {
    let polygons = PolygonArray::new(downcast(polygons, "a list of rings")?)?;
    let holes = lines_per_row(holes, listed, null_parts)?;
    let validity = rows_validity(&polygons.rows, holes.rows(), null_parts);
    let room = have_room_for_holes(&polygons, &holes);
    let (holes, validity) = match empty_exterior {
        EmptyExterior::Null => (
            holes,
            combine_validities_and(validity.as_ref(), Some(&room)),
        ),
        EmptyExterior::Keep => (holes.only_on_rows(&room), validity),
    };
    let added = polygons_with_added_holes(&polygons, Some(&holes), validity);
    Ok(GeometryArray::Polygon(validated_polygons(added, close)))
}
