//! Reversing the order of a geometry's vertices.
//! Implements the Python `reverse`.
//!
//! Every kernel copies the coordinates once, then reverses runs of them in place.
//! The list levels are shared with the input, unless the order of the parts changes.
//! A null row is reversed like any other, as its value is masked anyway.
//!
//! # Const generic flags
//!
//! ## `REVERSE_HOLES`
//!
//! [reverse_polygons] and [reverse_multi_polygons] are generic over `REVERSE_HOLES`,
//! the Python `reverse_holes`.
//!
//! ## `NESTED`
//!
//! [reverse_multi_line_strings] and [reverse_multi_polygons] are generic over `NESTED`,
//! the Python `nested`.

use std::ops::Range;

use geopolars_views::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
    ValidCoords, ValidList,
};
use polars_arrow::offset::Offsets;

use super::rings::offsets_of;

/// `coords` with the exterior of each of `polygons` reversed,
/// or every ring if `REVERSE_HOLES=true`.
/// Each item of `polygons` is the range of that polygon's rings in `rings`.
fn reverse_rings<const REVERSE_HOLES: bool>(
    polygons: impl Iterator<Item = Range<usize>> + Clone,
    rings: &ValidList,
    coords: &ValidCoords,
) -> ValidCoords {
    if REVERSE_HOLES {
        coords.with_ranges_reversed(rings.ranges())
    } else {
        let exteriors = polygons
            .filter(|polygon| !polygon.is_empty())
            .map(|polygon| rings.range(polygon.start));
        coords.with_ranges_reversed(exteriors)
    }
}

/// `parts` with the order of the parts of every row of `rows` reversed,
/// for when all items of each row are reversed as one.
fn parts_reversed_per_row(rows: &ValidList, parts: &ValidList) -> ValidList {
    let old: &[i64] = parts.offsets().buffer();
    let mut new = old.to_vec();
    for row in rows.ranges() {
        // Reversing the items `start..end` moves the boundary at `o` to `start + end - o`.
        let (start, end) = (old[row.start], old[row.end]);
        for k in 0..=row.len() {
            new[row.start + k] = start + end - old[row.end - k];
        }
    }
    // SAFETY: within a row, the new offsets grow as the old ones do,
    // and every row still starts and ends where it did.
    ValidList::from_offsets(unsafe { Offsets::new_unchecked(new) }.into())
}

/// `array` with the polygons of every row in reverse order.
/// The rings of each polygon keep their order and coordinates.
fn polygons_in_reverse_order(array: &MultiPolygonArray) -> MultiPolygonArray {
    let (rows, polygons, rings) = (
        array.rows.ignoring_validity(),
        &array.polygons,
        &array.rings,
    );
    let reordered = rows.ranges().flat_map(|row| row.rev());
    let ring_lengths = reordered
        .clone()
        .flat_map(|j| polygons.range(j).map(|ring| rings.range(ring).len()));
    MultiPolygonArray::from_parts(
        List::from_offsets(
            offsets_of(rows.ranges().map(|row| row.len())),
            array.rows.validity().cloned(),
        ),
        ValidList::from_offsets(offsets_of(
            reordered.clone().map(|j| polygons.range(j).len()),
        )),
        ValidList::from_offsets(offsets_of(ring_lengths)),
        array
            .coords
            .copy_ranges(reordered.map(|j| rings.span(polygons.range(j)))),
    )
}

pub fn reverse_line_strings(array: &LineStringArray) -> LineStringArray {
    let rows = array.rows.ignoring_validity().ranges();
    LineStringArray::from_parts(array.rows.clone(), array.coords.with_ranges_reversed(rows))
}

pub fn reverse_multi_points(array: &MultiPointArray) -> MultiPointArray {
    let rows = array.rows.ignoring_validity().ranges();
    MultiPointArray::from_parts(array.rows.clone(), array.coords.with_ranges_reversed(rows))
}

pub fn reverse_polygons<const REVERSE_HOLES: bool>(array: &PolygonArray) -> PolygonArray {
    let rows = array.rows.ignoring_validity().ranges();
    PolygonArray::from_parts(
        array.rows.clone(),
        array.rings.clone(),
        reverse_rings::<REVERSE_HOLES>(rows, &array.rings, &array.coords),
    )
}

pub fn reverse_multi_line_strings<const NESTED: bool>(
    array: &MultiLineStringArray,
) -> MultiLineStringArray {
    let (rows, lines) = (array.rows.ignoring_validity(), &array.lines);
    if NESTED {
        let spans = rows.ranges().map(|parts| lines.span(parts));
        MultiLineStringArray::from_parts(
            array.rows.clone(),
            parts_reversed_per_row(rows, lines),
            array.coords.with_ranges_reversed(spans),
        )
    } else {
        MultiLineStringArray::from_parts(
            array.rows.clone(),
            lines.clone(),
            array.coords.with_ranges_reversed(lines.ranges()),
        )
    }
}

/// With `NESTED=true`, the polygons are first reordered into a copy.
pub fn reverse_multi_polygons<const REVERSE_HOLES: bool, const NESTED: bool>(
    array: &MultiPolygonArray,
) -> MultiPolygonArray {
    if NESTED {
        return reverse_multi_polygons::<REVERSE_HOLES, false>(&polygons_in_reverse_order(array));
    }
    let polygons = array.polygons.ranges();
    MultiPolygonArray::from_parts(
        array.rows.clone(),
        array.polygons.clone(),
        array.rings.clone(),
        reverse_rings::<REVERSE_HOLES>(polygons, &array.rings, &array.coords),
    )
}
