//! Whether a linestring is a ring, or the parts of a multilinestring chain into one.
//! Implements the Python `is_ring`.
//! The polygon kernels serve the operations that build polygons out of rings.
//!
//! # Const generic flags
//!
//! See [same_place] for `Z`.

use geopolars_views::{
    is_ring, same_place, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
    ScalarMultiLineString, ScalarPolygon, Untrimmed, MIN_RING_VERTICES,
};
use polars_arrow::array::BooleanArray;

use crate::per_row::per_row_bool;
use crate::EmptyPolygonExterior;

pub fn line_strings_are_rings<const Z: bool>(array: &Untrimmed<LineStringArray>) -> BooleanArray {
    per_row_bool(array.rows(), |i| {
        is_ring::<Z>(array.coords(), array.rows().range(i))
    })
}

#[inline(always)]
fn parts_form_ring<const Z: bool>(multi: &ScalarMultiLineString) -> bool {
    let coords = multi.coords;
    let mut parts = multi.line_strings();
    let Some(first) = parts.next() else {
        return false;
    };
    if first.vertices.is_empty() {
        return false;
    }
    let mut joined_vertices = first.vertices.len();
    let mut end = first.vertices.end - 1;
    for part in parts {
        if part.vertices.is_empty() || !same_place::<Z>(coords, end, part.vertices.start) {
            return false;
        }
        joined_vertices += part.vertices.len() - 1;
        end = part.vertices.end - 1;
    }
    joined_vertices >= MIN_RING_VERTICES && same_place::<Z>(coords, first.vertices.start, end)
}

pub fn multi_line_strings_are_rings<const Z: bool>(
    array: &Untrimmed<MultiLineStringArray>,
) -> BooleanArray {
    per_row_bool(array.rows(), |i| parts_form_ring::<Z>(&array.get(i)))
}

// TODO: Not yet converted to bitmask pass, but is a minor candidate for it.
// Mark the rings that are rings or empty in a mask,
// and fold it with `rows_passing` (see the TODO at `polygon_shape_is_valid`).
#[inline(always)]
fn has_only_rings<const Z: bool>(
    polygon: &ScalarPolygon,
    allow_holes_in_empty_exterior: bool,
) -> bool {
    if polygon.has_empty_exterior() {
        if allow_holes_in_empty_exterior {
            return polygon
                .rings()
                .skip(1)
                .all(|hole| hole.vertices.is_empty() || is_ring::<Z>(hole.coords, hole.vertices));
        }
        return polygon.rings().all(|ring| ring.vertices.is_empty());
    }
    polygon
        .rings()
        .all(|ring| is_ring::<Z>(ring.coords, ring.vertices))
}

/// Whether every ring of each polygon is a ring, or every one is empty (the empty polygon).
/// With [EmptyPolygonExterior::Keep], an empty exterior may also have holes that are rings or empty.
/// Every other mode fails such a polygon,
/// as `"error"` and `"empty"` have already acted on it in [super::holes_added_to_polygons].
pub fn polygons_have_only_rings<const Z: bool>(
    array: &PolygonArray,
    empty_polygon_exterior: EmptyPolygonExterior,
) -> BooleanArray {
    let allow_holes_in_empty_exterior = empty_polygon_exterior == EmptyPolygonExterior::Keep;
    per_row_bool(&array.rows, |i| {
        has_only_rings::<Z>(&array.get(i), allow_holes_in_empty_exterior)
    })
}

/// Whether every polygon of each multipolygon has only rings, as [polygons_have_only_rings] asks.
pub fn multi_polygons_have_only_rings<const Z: bool>(array: &MultiPolygonArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array
            .get(i)
            .polygons()
            .all(|polygon| has_only_rings::<Z>(&polygon, false))
    })
}
