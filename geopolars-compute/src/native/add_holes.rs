//! Adding holes to polygons from another column.

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
///
/// The arguments must uphold the preconditions of [polygons_with_added_holes].
pub fn holes_added_to_polygons(
    polygons: &PolygonArray,
    holes: MaskedParts<MultiLineStringArray>,
    output_validity: Option<Bitmap>,
    empty_polygon_exterior: EmptyPolygonExterior,
    on_row_error: OnRowError,
) -> GeoComputeResult<PolygonArray> {
    let room = holes_added_to_polygons_have_room(polygons, &holes);
    let (holes, output_validity) = match empty_polygon_exterior {
        EmptyPolygonExterior::Keep => {
            // A polygon without rings would take its first hole as its exterior.
            let polygons = holes_added_to_polygons_with_empty_exteriors(polygons, &!&room);
            return Ok(polygons_with_added_holes(
                &polygons,
                Some(&holes),
                output_validity,
            ));
        }
        EmptyPolygonExterior::Empty => (holes.only_on_rows(&room), output_validity),
        mode @ (EmptyPolygonExterior::Null | EmptyPolygonExterior::Error) => {
            let with_room = combine_validities_and(output_validity.as_ref(), Some(&room));
            // A row that is null already has no exterior to complain about.
            if mode == EmptyPolygonExterior::Error
                && on_row_error == OnRowError::Raise
                && null_count(&with_room) > null_count(&output_validity)
            {
                return Err(GeoComputeError::HoleInEmptyExterior);
            }
            (holes, with_room)
        }
    };
    Ok(polygons_with_added_holes(
        polygons,
        Some(&holes),
        output_validity,
    ))
}

/// Gives an empty exterior to each polygon without rings that is set in `marked`,
/// so that the holes it gets next stay holes.
fn holes_added_to_polygons_with_empty_exteriors(
    polygons: &PolygonArray,
    marked: &Bitmap,
) -> PolygonArray {
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

/// Marks the polygons that can take their holes.
/// Only an empty exterior cannot, and only if it gets a hole with vertices that is set in the mask of `holes`.
fn holes_added_to_polygons_have_room(
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
