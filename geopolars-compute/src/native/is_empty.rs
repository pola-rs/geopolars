//! Whether a geometry is empty.
//! See `is_empty` in Python for the rules.
//!
//! # Const generic flags
//!
//! ## `CHECK_NESTED`
//!
//! Every kernel but the one of points is generic over `CHECK_NESTED`.
//! If `CHECK_NESTED=false`, a geometry is empty when it has no vertices.
//! This only reads the offsets.
//! If `CHECK_NESTED=true`, it is also empty when every vertex is the [empty point](is_empty_point).
//! This reads every vertex, so it is much more expensive.

use std::ops::Range;

use geopolars_core::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, Untrimmed, ValidCoords,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use super::coordinate_selection::is_empty_point;
use crate::per_row::per_row_bool;

/// Whether every one of `vertices` is the empty point, which also holds when there are none.
#[inline(always)]
fn only_empty_points(coords: &ValidCoords, mut vertices: Range<usize>) -> bool {
    vertices.all(|i| is_empty_point(coords, i))
}

// TODO: Not yet converted to bitmask pass, but is a good candidate for it.
// With `CHECK_NESTED=true`, mark the empty points in a mask,
// and fold it with `rows_passing` (as `has_repeats_in_succession` does).
// `rows_passing` needs trimmed lists, so the kernels would trim their `Untrimmed` views first.
#[inline(always)]
fn has_no_points<const CHECK_NESTED: bool>(coords: &ValidCoords, vertices: Range<usize>) -> bool {
    if CHECK_NESTED {
        only_empty_points(coords, vertices)
    } else {
        vertices.is_empty()
    }
}

pub fn points_are_empty(array: &PointArray) -> BooleanArray {
    let coords = array.coords.ignoring_validity();
    // A null point is tested too, but its value is masked by the validity.
    let values = Bitmap::from_iter((0..coords.len()).map(|i| is_empty_point(coords, i)));
    BooleanArray::new(
        ArrowDataType::Boolean,
        values,
        array.coords.validity().cloned(),
    )
}

pub fn line_strings_are_empty<const CHECK_NESTED: bool>(
    array: &Untrimmed<LineStringArray>,
) -> BooleanArray {
    per_row_bool(array.rows(), |i| {
        has_no_points::<CHECK_NESTED>(array.coords(), array.rows().range(i))
    })
}

pub fn multi_points_are_empty<const CHECK_NESTED: bool>(
    array: &Untrimmed<MultiPointArray>,
) -> BooleanArray {
    per_row_bool(array.rows(), |i| {
        has_no_points::<CHECK_NESTED>(array.coords(), array.rows().range(i))
    })
}

pub fn multi_line_strings_are_empty<const CHECK_NESTED: bool>(
    array: &Untrimmed<MultiLineStringArray>,
) -> BooleanArray {
    per_row_bool(array.rows(), |i| {
        has_no_points::<CHECK_NESTED>(array.coords(), array.lines().span(array.rows().range(i)))
    })
}

pub fn polygons_are_empty<const CHECK_NESTED: bool>(
    array: &Untrimmed<PolygonArray>,
) -> BooleanArray {
    per_row_bool(array.rows(), |i| {
        has_no_points::<CHECK_NESTED>(array.coords(), array.rings().span(array.rows().range(i)))
    })
}

pub fn multi_polygons_are_empty<const CHECK_NESTED: bool>(
    array: &Untrimmed<MultiPolygonArray>,
) -> BooleanArray {
    per_row_bool(array.rows(), |i| {
        let vertices = array
            .rings()
            .span(array.polygons().span(array.rows().range(i)));
        has_no_points::<CHECK_NESTED>(array.coords(), vertices)
    })
}
