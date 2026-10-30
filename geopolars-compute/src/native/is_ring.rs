//! Whether a linestring is a ring: at least [MIN_RING_VERTICES] vertices,
//! with its last at exactly the same place as its first.
//! A multilinestring is a ring when its parts chain into one (see [multi_line_strings_are_rings]).
//! A polygon has only rings when every ring is one, or every ring is empty.
//! m is a measure rather than a position, so it is not compared.
//!
//! # Const generic flags
//!
//! See [same_place] for `Z`.

use geopolars_core::{
    is_ring, same_place, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
    ScalarMultiLineString, ScalarPolygon, Untrimmed, MIN_RING_VERTICES,
};
use polars_arrow::array::BooleanArray;

use crate::per_row::per_row_bool;

pub fn line_strings_are_rings<const Z: bool>(array: &Untrimmed<LineStringArray>) -> BooleanArray {
    per_row_bool(array.rows(), |i| {
        is_ring::<Z>(array.coords(), array.rows().range(i))
    })
}

/// Whether the parts of `multi` chain into one ring.
/// each part ends where the next starts, and the last ends where the first starts.
/// Other than that same requirements as single linestring.
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

/// See [parts_form_ring] for what makes a ring.
pub fn multi_line_strings_are_rings<const Z: bool>(
    array: &Untrimmed<MultiLineStringArray>,
) -> BooleanArray {
    per_row_bool(array.rows(), |i| parts_form_ring::<Z>(&array.get(i)))
}

// TODO: Not yet converted to bitmask pass, but is a minor candidate for it.
// Mark the rings that are rings or empty in a mask,
// and fold it with `rows_passing` (see the TODO at `polygon_shape_is_valid`).
/// Whether every ring of `polygon` is a ring, or every one is empty (the empty polygon).
#[inline(always)]
fn has_only_rings<const Z: bool>(polygon: &ScalarPolygon) -> bool {
    if polygon.has_empty_exterior() {
        return polygon.rings().all(|ring| ring.vertices.is_empty());
    }
    polygon
        .rings()
        .all(|ring| is_ring::<Z>(ring.coords, ring.vertices))
}

/// See [has_only_rings].
pub fn polygons_have_only_rings<const Z: bool>(array: &PolygonArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| has_only_rings::<Z>(&array.get(i)))
}

/// Whether every polygon [has only rings](has_only_rings).
pub fn multi_polygons_have_only_rings<const Z: bool>(array: &MultiPolygonArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array
            .get(i)
            .polygons()
            .all(|polygon| has_only_rings::<Z>(&polygon))
    })
}
