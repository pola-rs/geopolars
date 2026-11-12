//! Whether a linestring or polygon is simple.
//! See `is_simple` in Python for the rules.
//!
//! GEOS only knows how lines run for finite x and y, so anything else is not simple.
//! A polygon is simple when each of its rings is, so every ring goes to GEOS as a linestring on its own.

use std::ops::Range;

use geopolars_views::{
    LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray, ScalarMultiLineString,
    ScalarPolygon, MIN_RING_VERTICES,
};
use geos::Geom;
use polars_arrow::array::BooleanArray;

use super::linestrings::line_string_to_geos;
use super::{axes, xy_is_finite_or_error, Axes, ToGeos};
use crate::per_row::try_per_row_bool;
use crate::GeoComputeResult;

fn is_xy_finite(axes: &Axes, vertices: Range<usize>) -> bool {
    let (x, y, _, _) = axes;
    xy_is_finite_or_error(&x[vertices.clone()], &y[vertices]).is_ok()
}

fn line_is_simple(axes: &Axes, vertices: Range<usize>) -> GeoComputeResult<bool> {
    if !is_xy_finite(axes, vertices.clone()) {
        return Ok(false);
    }
    match vertices.len() {
        // An empty line or a single segment can't cross itself.
        0 | 2 => Ok(true),
        // A single vertex is not a line, and GEOS can't build one.
        1 => Ok(false),
        _ => Ok(line_string_to_geos(axes, vertices)?.is_simple()?),
    }
}

/// Whether the ring through `vertices` is simple.
/// With fewer than 4 vertices, a ring that is not empty can only go back and forth.
fn ring_is_simple(axes: &Axes, vertices: Range<usize>) -> GeoComputeResult<bool> {
    if vertices.is_empty() {
        return Ok(true);
    }
    if vertices.len() < MIN_RING_VERTICES || !is_xy_finite(axes, vertices.clone()) {
        return Ok(false);
    }
    Ok(line_string_to_geos(axes, vertices)?.is_simple()?)
}

fn polygon_is_simple(axes: &Axes, polygon: &ScalarPolygon) -> GeoComputeResult<bool> {
    for j in polygon.parts.clone() {
        if !ring_is_simple(axes, polygon.rings.range(j))? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether `multi` is simple: every part is, and two parts only meet where both have an end.
fn multi_line_is_simple(multi: &ScalarMultiLineString) -> GeoComputeResult<bool> {
    let axes = axes(multi.coords);
    if !is_xy_finite(&axes, multi.lines.span(multi.parts.clone())) {
        return Ok(false);
    }
    // A single vertex is not a line, and GEOS can't build one.
    if multi.parts.clone().any(|j| multi.lines.range(j).len() == 1) {
        return Ok(false);
    }
    Ok(multi.to_geos()?.is_simple()?)
}

pub fn line_strings_are_simple(array: &LineStringArray) -> GeoComputeResult<BooleanArray> {
    let axes = axes(&array.coords);
    try_per_row_bool(&array.rows, |i| line_is_simple(&axes, array.rows.range(i)))
}

pub fn multi_line_strings_are_simple(
    array: &MultiLineStringArray,
) -> GeoComputeResult<BooleanArray> {
    try_per_row_bool(&array.rows, |i| multi_line_is_simple(&array.get(i)))
}

pub fn polygons_are_simple(array: &PolygonArray) -> GeoComputeResult<BooleanArray> {
    let axes = axes(&array.coords);
    try_per_row_bool(&array.rows, |i| polygon_is_simple(&axes, &array.get(i)))
}

pub fn multi_polygons_are_simple(array: &MultiPolygonArray) -> GeoComputeResult<BooleanArray> {
    let axes = axes(&array.coords);
    try_per_row_bool(&array.rows, |i| {
        for polygon in array.get(i).polygons() {
            if !polygon_is_simple(&axes, &polygon)? {
                return Ok(false);
            }
        }
        Ok(true)
    })
}
