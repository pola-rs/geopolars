//! Whether a linestring crosses or touches itself, in x and y, found by rsgeo's sweep.
//! A polygon also intersects itself when two of its holes overlap.

use std::convert::Infallible;

use ::rsgeo::algorithm::sweep::Cross;
use ::rsgeo::{Coord, Intersections, Line, LineIntersection, PreparedGeometry, Relate};
use geopolars_views::{
    LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray, ScalarLineString,
    ScalarPolygon,
};
use polars_arrow::array::{BooleanArray, ListArray};

use super::rings_overlap::overlap;
use super::{
    per_part_optional_bool, per_row_optional_bool, ring_is_finite, to_rsgeo_ring, xy_is_finite,
};
use crate::{row_or_null, GeoComputeResult};

/// One segment of a linestring
#[derive(Clone, Copy)]
struct Segment {
    index: usize,
    line: Line<f64>,
}

impl Cross for Segment {
    type Scalar = f64;

    fn line(&self) -> Line<f64> {
        self.line
    }
}

/// Whether `a` and `b` lie at exactly the same place in x and y.
#[inline(always)]
fn same_xy(a: Coord<f64>, b: Coord<f64>) -> bool {
    a == b
}

/// Whether the line through the `n` vertices `vertex(0)` up to `vertex(n - 1)`
/// crosses or touches itself.
///
/// A segment whose ends lie at the same place is skipped.
/// Two segments in a row (and the last and the first, when the line closes)
/// may share their vertex, but not overlap.
/// any other two may not touch at all.
#[inline(always)]
fn self_intersect(n: usize, vertex: impl Fn(usize) -> Coord<f64>) -> bool {
    let vertex = &vertex;
    let ends = || {
        (1..n)
            .map(move |i| (vertex(i - 1), vertex(i)))
            .filter(|&(a, b)| !same_xy(a, b))
    };
    // Counted rather than collected: the sweep collects the segments itself.
    let segments = ends().count();
    if segments < 2 {
        return false;
    }
    let closed = same_xy(vertex(0), vertex(n - 1));
    let adjacent = |a: &Segment, b: &Segment| {
        let (low, high) = (a.index.min(b.index), a.index.max(b.index));
        high - low == 1 || (closed && low == 0 && high == segments - 1)
    };
    let segments = ends().enumerate().map(|(index, (a, b))| Segment {
        index,
        line: Line::new(a, b),
    });
    Intersections::from_iter(segments)
        .any(|(a, b, hit)| !adjacent(&a, &b) || matches!(hit, LineIntersection::Collinear { .. }))
}

/// Whether the line through `vertices` crosses or touches itself.
/// See [self_intersect].
pub fn vertices_self_intersect(vertices: &[Coord<f64>]) -> bool {
    self_intersect(vertices.len(), |i| vertices[i])
}

/// [self_intersect] for `line`, which has to be finite in x and y.
#[inline(always)]
pub(super) fn finite_line_self_intersects(line: &ScalarLineString) -> bool {
    let x = &line.coords.x()[line.vertices.clone()];
    let y = &line.coords.y()[line.vertices.clone()];
    self_intersect(x.len(), |i| Coord { x: x[i], y: y[i] })
}

/// [self_intersect] for `line`.
/// `None` if one of its vertices is not finite in x or y.
#[inline(always)]
fn line_self_intersects(line: &ScalarLineString) -> Option<bool> {
    xy_is_finite(line).then(|| finite_line_self_intersects(line))
}

/// Whether `polygon` intersects itself:
/// one of its rings crosses itself, or two of its holes (every ring after the first) overlap.
/// `None` if one of its vertices is not finite in x or y.
/// Errors if one of its rings is not a ring.
fn polygon_self_intersects(polygon: &ScalarPolygon) -> GeoComputeResult<Option<bool>> {
    // Every ring is checked before any is swept,
    // so a ring crossing itself does not hide a later one that is not finite, or not a ring.
    for ring in polygon.rings() {
        if !ring_is_finite(&ring)? {
            return Ok(None);
        }
    }
    if polygon
        .rings()
        .any(|ring| finite_line_self_intersects(&ring))
    {
        return Ok(Some(true));
    }
    // Only with two holes is there anything to relate, so only then are they copied for rsgeo.
    if polygon.parts.len() < 3 {
        return Ok(Some(false));
    }
    // Prepare every hole once, as it is related to every other.
    let holes: Vec<_> = polygon
        .rings()
        .skip(1)
        .map(|hole| PreparedGeometry::from(to_rsgeo_ring(&hole)))
        .collect();
    Ok(Some(holes.iter().enumerate().any(|(i, hole)| {
        holes[i + 1..]
            .iter()
            .any(|other| overlap::<false>(&hole.relate(other)))
    })))
}

/// Whether each linestring intersects itself, null where it is not finite in x or y.
pub fn line_strings_self_intersect(array: &LineStringArray) -> BooleanArray {
    let Ok(out) = per_row_optional_bool(&array.rows, |i| {
        Ok::<_, Infallible>(line_self_intersects(&array.get(i)))
    });
    out
}

/// [line_strings_self_intersect], as a list per row of a chunk, one per part.
pub fn multi_line_strings_self_intersect(array: &MultiLineStringArray) -> ListArray<i64> {
    let Ok(out) = per_part_optional_bool(&array.rows, &array.lines, |part| {
        Ok::<_, Infallible>(line_self_intersects(&ScalarLineString {
            coords: &array.coords,
            vertices: array.lines.range(part),
        }))
    });
    out
}

/// Whether each polygon [intersects itself](polygon_self_intersects).
/// If `ignore_per_row_errors=true`, a polygon with a ring that is not one is null;
/// otherwise that is the error.
pub fn polygons_self_intersect(
    array: &PolygonArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<BooleanArray> {
    per_row_optional_bool(&array.rows, |i| {
        row_or_null(
            polygon_self_intersects(&array.get(i)),
            ignore_per_row_errors,
        )
    })
}

/// [polygons_self_intersect], as a list per row of a chunk, one per part.
/// If `ignore_per_row_errors=true`, only the part with a ring that is not one is null.
pub fn multi_polygons_self_intersect(
    array: &MultiPolygonArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<ListArray<i64>> {
    per_part_optional_bool(&array.rows, &array.polygons, |part| {
        let polygon = ScalarPolygon {
            rings: &array.rings,
            coords: &array.coords,
            parts: array.polygons.range(part),
        };
        row_or_null(polygon_self_intersects(&polygon), ignore_per_row_errors)
    })
}
