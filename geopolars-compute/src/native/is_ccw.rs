//! Whether a geometry runs counter-clockwise, and making it so.
//! Implements the Python `is_ccw` and `to_ccw`.
//!
//! # Const generic flags
//!
//! ## `HOLES_CW`
//!
//! The polygon and multipolygon kernels are generic over `HOLES_CW`, the Python `holes_cw`.

use std::ops::Range;

use geopolars_views::{
    LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray, ScalarLineString,
    ScalarPolygon, Untrimmed, ValidCoords, ValidList,
};
use polars_arrow::array::BooleanArray;

use super::area::twice_signed_area;
use crate::per_row::per_row_bool;

/// Whether `ring` runs counter-clockwise, or clockwise if `CW=true`,
/// by the rules of the Python `is_ccw`.
/// An empty ring therefore gives `true` for either `CW`.
#[inline(always)]
fn ring_runs<const CW: bool>(ring: &ScalarLineString) -> bool {
    if ring.vertices.is_empty() {
        return true;
    }
    let winding = twice_signed_area(ring);
    if CW {
        winding < 0.0
    } else {
        winding > 0.0
    }
}

#[inline(always)]
fn polygon_is_ccw<const HOLES_CW: bool>(polygon: &ScalarPolygon) -> bool {
    let mut rings = polygon.rings();
    let Some(exterior) = rings.next() else {
        return true;
    };
    ring_runs::<false>(&exterior) && rings.all(|hole| ring_runs::<HOLES_CW>(&hole))
}

pub fn line_strings_are_ccw(array: &Untrimmed<LineStringArray>) -> BooleanArray {
    per_row_bool(array.rows(), |i| ring_runs::<false>(&array.get(i)))
}

pub fn polygons_are_ccw<const HOLES_CW: bool>(array: &Untrimmed<PolygonArray>) -> BooleanArray {
    per_row_bool(array.rows(), |i| polygon_is_ccw::<HOLES_CW>(&array.get(i)))
}

pub fn multi_line_strings_are_ccw(array: &Untrimmed<MultiLineStringArray>) -> BooleanArray {
    per_row_bool(array.rows(), |i| {
        array
            .get(i)
            .line_strings()
            .all(|line| ring_runs::<false>(&line))
    })
}

pub fn multi_polygons_are_ccw<const HOLES_CW: bool>(
    array: &Untrimmed<MultiPolygonArray>,
) -> BooleanArray {
    per_row_bool(array.rows(), |i| {
        array
            .get(i)
            .polygons()
            .all(|polygon| polygon_is_ccw::<HOLES_CW>(&polygon))
    })
}

/// Which of `lines` run clockwise, so have to be reversed to run counter-clockwise.
/// Collected, as [ValidCoords::with_ranges_reversed] goes over them once per axis.
fn lines_to_flip(
    lines: impl Iterator<Item = Range<usize>>,
    coords: &ValidCoords,
) -> Vec<Range<usize>> {
    lines
        .filter(|vertices| {
            ring_runs::<true>(&ScalarLineString {
                coords,
                vertices: vertices.clone(),
            })
        })
        .collect()
}

// TODO: Not yet converted to bitmask pass, but is a good candidate for it.
// Mark the rings that run the wrong way in a mask, which needs the polygons to tell exteriors from holes.
// `is_ccw` then folds it with `rows_passing`, and `to_ccw` reverses the marked rings.
// This also removes the second copy of the exterior and hole rule, in `polygon_is_ccw`.
// First decide what to do with a ring that encloses nothing.
// It is not CCW now, yet `to_ccw` does not flip it.
/// Which rings of `polygons` have to be reversed to [run counter-clockwise](polygon_is_ccw).
/// Each item of `polygons` is the range of that polygon's rings in `rings`.
/// Collected, as [ValidCoords::with_ranges_reversed] goes over them once per axis.
fn rings_to_flip<const HOLES_CW: bool>(
    polygons: impl Iterator<Item = Range<usize>>,
    rings: &ValidList,
    coords: &ValidCoords,
) -> Vec<Range<usize>> {
    polygons
        .flat_map(|polygon| {
            let exterior = polygon.start;
            polygon.map(move |ring| (ring == exterior, rings.range(ring)))
        })
        .filter(|(is_exterior, vertices)| {
            let ring = ScalarLineString {
                coords,
                vertices: vertices.clone(),
            };
            // A ring that runs the way it should not has to be flipped.
            if *is_exterior || !HOLES_CW {
                ring_runs::<true>(&ring)
            } else {
                ring_runs::<false>(&ring)
            }
        })
        .map(|(_, vertices)| vertices)
        .collect()
}

pub fn line_strings_to_ccw(array: &LineStringArray) -> LineStringArray {
    let rows = array.rows.ignoring_validity().ranges();
    let flipped = lines_to_flip(rows, &array.coords);
    LineStringArray::from_parts(
        array.rows.clone(),
        array.coords.with_ranges_reversed(flipped.into_iter()),
    )
}

pub fn polygons_to_ccw<const HOLES_CW: bool>(array: &PolygonArray) -> PolygonArray {
    let rows = array.rows.ignoring_validity().ranges();
    let flipped = rings_to_flip::<HOLES_CW>(rows, &array.rings, &array.coords);
    PolygonArray::from_parts(
        array.rows.clone(),
        array.rings.clone(),
        array.coords.with_ranges_reversed(flipped.into_iter()),
    )
}

pub fn multi_line_strings_to_ccw(array: &MultiLineStringArray) -> MultiLineStringArray {
    let flipped = lines_to_flip(array.lines.ranges(), &array.coords);
    MultiLineStringArray::from_parts(
        array.rows.clone(),
        array.lines.clone(),
        array.coords.with_ranges_reversed(flipped.into_iter()),
    )
}

pub fn multi_polygons_to_ccw<const HOLES_CW: bool>(array: &MultiPolygonArray) -> MultiPolygonArray {
    let polygons = array.polygons.ranges();
    let flipped = rings_to_flip::<HOLES_CW>(polygons, &array.rings, &array.coords);
    MultiPolygonArray::from_parts(
        array.rows.clone(),
        array.polygons.clone(),
        array.rings.clone(),
        array.coords.with_ranges_reversed(flipped.into_iter()),
    )
}
