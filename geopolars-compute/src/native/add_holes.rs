//! Adding holes to polygons from another column.
//! Implements the Python `add_holes`, up to closing and checking the rings.

use geopolars_views::{List, MultiLineStringArray, PolygonArray, ValidList};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;

use super::rings::{offsets_of, polygons_with_added_holes};
use crate::inner_nulls::MaskedParts;
use crate::{EmptyPolygonExterior, GeoComputeError, GeoComputeResult, OnRowError};

fn null_count(validity: &Option<Bitmap>) -> usize {
    validity.as_ref().map_or(0, Bitmap::unset_bits)
}

/// Adds `holes` to `polygons` as [polygons_with_added_holes] does,
/// treating a polygon with an empty exterior as `empty_polygon_exterior` asks.
/// Implements the Python `add_holes`, up to closing and checking the rings.
pub fn holes_added_to_polygons(
    polygons: &PolygonArray,
    holes: MaskedParts<MultiLineStringArray>,
    validity: Option<Bitmap>,
    empty_polygon_exterior: EmptyPolygonExterior,
    on_row_error: OnRowError,
) -> GeoComputeResult<PolygonArray> {
    let room = have_room_for_holes(polygons, &holes);
    let (holes, validity) = match empty_polygon_exterior {
        EmptyPolygonExterior::Keep => {
            // A polygon without rings would take its first hole as its exterior.
            let polygons = polygons_with_empty_exteriors(polygons, &!&room);
            return Ok(polygons_with_added_holes(&polygons, Some(&holes), validity));
        }
        EmptyPolygonExterior::Empty => (holes.only_on_rows(&room), validity),
        mode @ (EmptyPolygonExterior::Null | EmptyPolygonExterior::Error) => {
            let with_room = combine_validities_and(validity.as_ref(), Some(&room));
            // A row that is null already has no exterior to complain about.
            if mode == EmptyPolygonExterior::Error
                && on_row_error == OnRowError::Raise
                && null_count(&with_room) > null_count(&validity)
            {
                return Err(GeoComputeError::HoleInEmptyExterior);
            }
            (holes, with_room)
        }
    };
    Ok(polygons_with_added_holes(polygons, Some(&holes), validity))
}

/// `polygons`, with an empty exterior for each polygon without rings set in `marked`,
/// so that the holes it is given next stay holes.
fn polygons_with_empty_exteriors(polygons: &PolygonArray, marked: &Bitmap) -> PolygonArray {
    let (rows, rings) = (&polygons.rows, &polygons.rings);
    let gets_exterior =
        |i: usize| marked.get_bit(i) && rows.is_valid(i) && rows.range(i).is_empty();
    let ring_counts = (0..rows.len()).map(|i| match gets_exterior(i) {
        true => 1,
        false => rows.range(i).len(),
    });
    let ring_lengths = (0..rows.len()).flat_map(|i| {
        let exterior = gets_exterior(i).then_some(0);
        exterior
            .into_iter()
            .chain(rows.range(i).map(|j| rings.range(j).len()))
    });
    PolygonArray::from_parts(
        List::from_offsets(offsets_of(ring_counts), rows.validity().cloned()),
        ValidList::from_offsets(offsets_of(ring_lengths)),
        polygons.coords.clone(),
    )
}

/// Marks which polygons can take the holes on the same row of `holes`,
/// as `empty_polygon_exterior` of the Python `add_holes` describes.
/// Only the holes set in the mask of `holes` count towards this.
fn have_room_for_holes(
    polygons: &PolygonArray,
    holes: &MaskedParts<MultiLineStringArray>,
) -> Bitmap {
    let (rows, lines) = (&holes.view.rows, &holes.view.lines);
    let is_kept = |line: usize| holes.keep.as_ref().is_none_or(|keep| keep.get_bit(line));
    (0..polygons.rows.len())
        .map(|i| {
            !polygons.rows.is_valid(i)
                || !polygons.get(i).has_empty_exterior()
                || !rows.is_valid(i)
                || rows
                    .range(i)
                    .all(|line| !is_kept(line) || lines.range(line).is_empty())
        })
        .collect()
}
