//! How many coordinates a geometry is made of.
//! Implements the Python `count_coordinates`.
//!
//! # Const generic flags
//!
//! ## `INCLUDE_EMPTY`
//!
//! Every function here is generic over `INCLUDE_EMPTY`.
//! If `INCLUDE_EMPTY=false`, the [empty point](is_empty_point) does not count.
//! If `INCLUDE_EMPTY=true`, it counts like any other coordinate.

use std::ops::Range;

use geopolars_views::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, ScalarPolygon, Untrimmed, ValidCoords,
};
use polars_arrow::array::PrimitiveArray;

use super::coordinate_selection::is_empty_point;
use crate::per_row::per_row_count;

#[inline(always)]
fn count_vertices<const INCLUDE_EMPTY: bool>(coords: &ValidCoords, vertices: Range<usize>) -> u32 {
    if INCLUDE_EMPTY {
        vertices.len() as u32
    } else {
        vertices.filter(|&i| !is_empty_point(coords, i)).count() as u32
    }
}

/// [count_vertices], without the last vertex, which repeats the first.
#[inline(always)]
fn count_ring_vertices<const INCLUDE_EMPTY: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
) -> u32 {
    let open = vertices.start..vertices.end.saturating_sub(1).max(vertices.start);
    count_vertices::<INCLUDE_EMPTY>(coords, open)
}

#[inline(always)]
fn count_polygon_vertices<const INCLUDE_EMPTY: bool>(polygon: &ScalarPolygon) -> u32 {
    polygon
        .rings()
        .map(|ring| count_ring_vertices::<INCLUDE_EMPTY>(ring.coords, ring.vertices))
        .sum()
}

pub fn point_coordinate_counts<const INCLUDE_EMPTY: bool>(
    array: &PointArray,
) -> PrimitiveArray<u32> {
    let coords = array.coords.ignoring_validity();
    // A null point is counted too: its count is masked by the validity.
    let counts = (0..coords.len())
        .map(|i| count_vertices::<INCLUDE_EMPTY>(coords, i..i + 1))
        .collect();
    PrimitiveArray::from_vec(counts).with_validity(array.coords.validity().cloned())
}

pub fn line_string_coordinate_counts<const INCLUDE_EMPTY: bool>(
    array: &Untrimmed<LineStringArray>,
) -> PrimitiveArray<u32> {
    per_row_count(array.rows(), |i| {
        count_vertices::<INCLUDE_EMPTY>(array.coords(), array.rows().range(i))
    })
}

pub fn multi_point_coordinate_counts<const INCLUDE_EMPTY: bool>(
    array: &Untrimmed<MultiPointArray>,
) -> PrimitiveArray<u32> {
    per_row_count(array.rows(), |i| {
        count_vertices::<INCLUDE_EMPTY>(array.coords(), array.rows().range(i))
    })
}

pub fn multi_line_string_coordinate_counts<const INCLUDE_EMPTY: bool>(
    array: &Untrimmed<MultiLineStringArray>,
) -> PrimitiveArray<u32> {
    per_row_count(array.rows(), |i| {
        count_vertices::<INCLUDE_EMPTY>(array.coords(), array.lines().span(array.rows().range(i)))
    })
}

pub fn polygon_coordinate_counts<const INCLUDE_EMPTY: bool>(
    array: &Untrimmed<PolygonArray>,
) -> PrimitiveArray<u32> {
    per_row_count(array.rows(), |i| {
        count_polygon_vertices::<INCLUDE_EMPTY>(&array.get(i))
    })
}

pub fn multi_polygon_coordinate_counts<const INCLUDE_EMPTY: bool>(
    array: &Untrimmed<MultiPolygonArray>,
) -> PrimitiveArray<u32> {
    per_row_count(array.rows(), |i| {
        array
            .get(i)
            .polygons()
            .map(|polygon| count_polygon_vertices::<INCLUDE_EMPTY>(&polygon))
            .sum()
    })
}
