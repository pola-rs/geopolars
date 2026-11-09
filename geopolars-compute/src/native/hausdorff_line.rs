//! The line between two geometries that spans their Hausdorff distance.
//! See Python `hausdorff_line_between` for the rules.
//!
//! Every vertex of each geometry is measured to the nearest point on the parts of the other, in x and y.
//! The furthest of these over both directions gives the line, as in the discrete Hausdorff distance of GEOS.
//! It is a brute-force search: every vertex against every segment of the other geometry,
//! except that a vertex stops being measured once it can no longer be the furthest.
//!
//! # Why not GEOS
//!
//! GEOS has the same search as `GEOSHausdorffDistanceWithPoints`, but we cannot use it here.
//! The `geos` crate does not wrap it, nor give out the raw pointers to call it ourselves.
//! Its points come in the order of whichever direction gave the larger distance,
//! not of its inputs, so the line would not reliably run from `a` to `b`.
//! It also drops z and m, and every row would first be copied into a GEOS geometry.
//! GEOS searches by brute force too (without stopping early),
//! so there is no tested algorithm here that we would have to re-implement.
//!
//! # Const generic flags
//!
//! ## `SKIP`
//!
//! The search is generic over `SKIP`, which is set when [HausdorffOptions] lets it leave out
//! the empty point, other coordinates that are not finite, or both.
//! If `SKIP=true`, it leaves out every coordinate that is not finite in the axes that count,
//! without telling those apart.
//! The options are only told apart by [Side::ensure_row_passes],
//! which fails a row holding a coordinate they do not let the search leave out,
//! before the search sees that row.

// PERF: There's plenty of opportunity to make stuff a little faster here.

use std::ops::Range;

use geopolars_core::{GeoDimension, GeometryArray, LineStringArray, List, ValidCoords, ValidList};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::Offsets;

use super::coordinate_selection::{is_empty_point, is_non_finite_value};
use super::{selected_coords, CoordinateSelection, NonFinite};
use crate::passing::rows_passing;
use crate::{GeoComputeError, GeoComputeResult};

/// Which coordinates the search leaves out.
/// Where neither `skip_empty` nor `skip_non_finite` lets it leave a coordinate out, its row fails.
/// `ignore_z` and `ignore_m` are as in [NonFinite].
#[derive(Debug, Clone, Copy, Default)]
pub struct HausdorffOptions {
    pub skip_empty: bool,
    pub skip_non_finite: bool,
    pub ignore_z: bool,
    pub ignore_m: bool,
}

impl HausdorffOptions {
    fn skip(self) -> bool {
        self.skip_empty || self.skip_non_finite
    }
}

/// How the rows of a geometry split into parts, level by level.
enum Levels<'a> {
    Points,
    MultiPoints {
        rows: &'a ValidList,
    },
    LineStrings {
        rows: &'a ValidList,
    },
    /// Polygons and multilinestrings, whose rows hold lists of vertices.
    Parts {
        rows: &'a ValidList,
        parts: &'a ValidList,
    },
    MultiPolygons {
        rows: &'a ValidList,
        polygons: &'a ValidList,
        rings: &'a ValidList,
    },
}

/// The parts of one row, as the range of vertices of each.
#[derive(Clone)]
enum RowParts<'a> {
    /// Every vertex is a part of its own.
    Vertices(Range<usize>),
    /// The vertices of each list.
    Lists(&'a ValidList, Range<usize>),
}

impl Iterator for RowParts<'_> {
    type Item = Range<usize>;

    #[inline(always)]
    fn next(&mut self) -> Option<Range<usize>> {
        match self {
            RowParts::Vertices(vertices) => vertices.next().map(|i| i..i + 1),
            RowParts::Lists(lists, indices) => indices.next().map(|j| lists.range(j)),
        }
    }
}

/// Where a point of a line lies on a geometry: `t` of the way from vertex `start` to vertex `end`.
#[derive(Clone, Copy)]
struct OnPart {
    start: usize,
    end: usize,
    t: f64,
}

impl OnPart {
    fn vertex(i: usize) -> Self {
        Self {
            start: i,
            end: i,
            t: 0.0,
        }
    }
}

/// The furthest one geometry's vertex gets from the other: a squared distance in x and y,
/// from `vertex` to the nearest point `nearest` on the other.
#[derive(Clone, Copy)]
struct Furthest {
    distance_squared: f64,
    vertex: usize,
    nearest: OnPart,
}

/// One side of the Hausdorff lines, with the coordinates that fail their row.
struct Side<'a> {
    coords: &'a ValidCoords,
    levels: Levels<'a>,
    /// The z and m values that count towards a coordinate being finite.
    counted_z: Option<&'a [f64]>,
    counted_m: Option<&'a [f64]>,
    /// The coordinates that fail their row, and the rows that hold none of them.
    /// `None` when there are none.
    failing: Option<(Bitmap, Option<Bitmap>)>,
}

impl<'a> Side<'a> {
    fn new(array: &'a GeometryArray, options: HausdorffOptions) -> Self {
        let (coords, levels) = match array {
            GeometryArray::Point(array) => (array.coords.ignoring_validity(), Levels::Points),
            GeometryArray::MultiPoint(array) => (
                &array.coords,
                Levels::MultiPoints {
                    rows: array.rows.ignoring_validity(),
                },
            ),
            GeometryArray::LineString(array) => (
                &array.coords,
                Levels::LineStrings {
                    rows: array.rows.ignoring_validity(),
                },
            ),
            GeometryArray::Polygon(array) => (
                &array.coords,
                Levels::Parts {
                    rows: array.rows.ignoring_validity(),
                    parts: &array.rings,
                },
            ),
            GeometryArray::MultiLineString(array) => (
                &array.coords,
                Levels::Parts {
                    rows: array.rows.ignoring_validity(),
                    parts: &array.lines,
                },
            ),
            GeometryArray::MultiPolygon(array) => (
                &array.coords,
                Levels::MultiPolygons {
                    rows: array.rows.ignoring_validity(),
                    polygons: &array.polygons,
                    rings: &array.rings,
                },
            ),
        };
        let mut side = Self {
            coords,
            levels,
            counted_z: coords.z().filter(|_| !options.ignore_z),
            counted_m: coords.m().filter(|_| !options.ignore_m),
            failing: None,
        };
        side.failing = side.failing_coords(options);
        side
    }

    fn failing_coords(&self, options: HausdorffOptions) -> Option<(Bitmap, Option<Bitmap>)> {
        let non_finite = NonFinite {
            select_non_finite: !options.skip_non_finite,
            // An empty point row is an empty geometry, rather than one holding an empty point.
            select_empty: !options.skip_empty && !matches!(self.levels, Levels::Points),
            ignore_z: options.ignore_z,
            ignore_m: options.ignore_m,
        };
        if !non_finite.select_non_finite && !non_finite.select_empty {
            return None;
        }
        let selection = CoordinateSelection {
            non_finite: Some(non_finite),
            bounds: None,
        };
        let failing = selected_coords(self.coords, &selection)?;
        if failing.set_bits() == 0 {
            return None;
        }
        let passing = Some(!&failing);
        let rows = match self.levels {
            Levels::Points => passing,
            Levels::MultiPoints { rows } | Levels::LineStrings { rows } => {
                rows_passing(&[rows], vec![None, passing])
            }
            Levels::Parts { rows, parts } => {
                rows_passing(&[rows, parts], vec![None, None, passing])
            }
            Levels::MultiPolygons {
                rows,
                polygons,
                rings,
            } => rows_passing(&[rows, polygons, rings], vec![None, None, None, passing]),
        };
        Some((failing, rows))
    }

    #[inline(always)]
    fn parts(&self, row: usize) -> RowParts<'a> {
        match self.levels {
            Levels::Points if is_empty_point(self.coords, row) => RowParts::Vertices(row..row),
            Levels::Points => RowParts::Vertices(row..row + 1),
            Levels::MultiPoints { rows } => RowParts::Vertices(rows.range(row)),
            Levels::LineStrings { rows } => RowParts::Lists(rows, row..row + 1),
            Levels::Parts { rows, parts } => RowParts::Lists(parts, rows.range(row)),
            Levels::MultiPolygons {
                rows,
                polygons,
                rings,
            } => RowParts::Lists(rings, polygons.span(rows.range(row))),
        }
    }

    #[inline(always)]
    fn has_no_vertices(&self, row: usize) -> bool {
        self.parts(row).flatten().next().is_none()
    }

    /// Fails if `row` holds a coordinate the search cannot leave out.
    fn ensure_row_passes(&self, row: usize) -> GeoComputeResult<()> {
        let Some((failing, rows)) = &self.failing else {
            return Ok(());
        };
        if rows.as_ref().is_none_or(|rows| rows.get_bit(row)) {
            return Ok(());
        }
        let i = self
            .parts(row)
            .flatten()
            .find(|&i| failing.get_bit(i))
            .expect("a failing coordinate in a row that does not pass");
        let coords = self.coords;
        Err(GeoComputeError::NotFiniteCoordinate {
            x: coords.x()[i],
            y: coords.y()[i],
            z: coords.z().map(|z| z[i]),
            m: coords.m().map(|m| m[i]),
        })
    }

    #[inline(always)]
    fn is_searched<const SKIP: bool>(&self, i: usize) -> bool {
        !SKIP
            || !is_non_finite_value(
                self.coords.x()[i],
                self.coords.y()[i],
                self.counted_z.map(|z| z[i]),
                self.counted_m.map(|m| m[i]),
            )
    }

    #[inline(always)]
    fn xy(&self, i: usize) -> (f64, f64) {
        (self.coords.x()[i], self.coords.y()[i])
    }
}

/// The point on segment `start`-`end` nearest to `(px, py)`, and its squared distance.
#[inline(always)]
fn nearest_on_segment(
    side: &Side,
    start: usize,
    end: usize,
    (px, py): (f64, f64),
) -> (f64, OnPart) {
    let ((x0, y0), (x1, y1)) = (side.xy(start), side.xy(end));
    let (dx, dy) = (x1 - x0, y1 - y0);
    let length_squared = dx * dx + dy * dy;
    let t = if length_squared > 0.0 {
        (((px - x0) * dx + (py - y0) * dy) / length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (ex, ey) = (px - (x0 + t * dx), py - (y0 + t * dy));
    (ex * ex + ey * ey, OnPart { start, end, t })
}

#[inline(always)]
fn keep_nearer(nearest: &mut Option<(f64, OnPart)>, candidate: (f64, OnPart)) {
    if nearest.is_none_or(|(distance_squared, _)| candidate.0 < distance_squared) {
        *nearest = Some(candidate);
    }
}

/// The point on row `row` of `side` nearest to `point`, and its squared distance.
/// `None` if the row has no vertex to search.
/// It stops at the first point at most `bound` (squared) away, as the caller has no use for a nearer one.
#[inline(always)]
fn nearest_on_row<const SKIP: bool>(
    side: &Side,
    row: usize,
    point: (f64, f64),
    bound: f64,
) -> Option<(f64, OnPart)> {
    let mut nearest = None;
    for part in side.parts(row) {
        let mut vertices = part.filter(|&i| side.is_searched::<SKIP>(i));
        let Some(first) = vertices.next() else {
            continue;
        };
        let mut previous = first;
        for vertex in vertices {
            let candidate = nearest_on_segment(side, previous, vertex, point);
            keep_nearer(&mut nearest, candidate);
            if candidate.0 <= bound {
                return nearest;
            }
            previous = vertex;
        }
        if previous == first {
            let (x, y) = side.xy(first);
            let (ex, ey) = (point.0 - x, point.1 - y);
            let candidate = (ex * ex + ey * ey, OnPart::vertex(first));
            keep_nearer(&mut nearest, candidate);
            if candidate.0 <= bound {
                return nearest;
            }
        }
    }
    nearest
}

/// The vertex of row `row` of `from` furthest from row `to_row` of `to`, if it is further than `floor` (squared).
/// `None` if no vertex is, or if either row has no vertex to search.
///
/// A vertex stops being measured once it lies at most as far as the furthest so far, as it can then no longer win.
/// It is first measured to the segment (or vertex) that held the previous vertex's nearest point,
/// which often settles that at once.
/// The vertex that wins is therefore always measured in full.
fn furthest<const SKIP: bool>(
    from: &Side,
    row: usize,
    to: &Side,
    to_row: usize,
    floor: f64,
) -> Option<Furthest> {
    let mut furthest: Option<Furthest> = None;
    let mut previous_nearest: Option<OnPart> = None;
    for vertex in from.parts(row).flatten() {
        if !from.is_searched::<SKIP>(vertex) {
            continue;
        }
        let bound = furthest.map_or(floor, |furthest| furthest.distance_squared);
        let point = from.xy(vertex);
        if previous_nearest
            .is_some_and(|on| nearest_on_segment(to, on.start, on.end, point).0 <= bound)
        {
            continue;
        }
        let (distance_squared, nearest) = nearest_on_row::<SKIP>(to, to_row, point, bound)?;
        previous_nearest = Some(nearest);
        // Strictly further, so that a tie goes to the first, as in GEOS.
        if distance_squared > bound {
            furthest = Some(Furthest {
                distance_squared,
                vertex,
                nearest,
            });
        }
    }
    furthest
}

/// The lines being built: no coordinates for an empty or null row, two for any other.
struct Lines {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Option<Vec<f64>>,
    m: Option<Vec<f64>>,
    offsets: Offsets<i64>,
    failed: Vec<usize>,
}

/// `v0`, `t` of the way to `v1`, without taking a value it does not reach (which may be NaN).
#[inline(always)]
fn interpolated(v0: f64, v1: f64, t: f64) -> f64 {
    if t == 0.0 {
        v0
    } else if t == 1.0 {
        v1
    } else {
        v0 + t * (v1 - v0)
    }
}

impl Lines {
    fn new(rows: usize, dim: GeoDimension) -> Self {
        let axis = || Vec::with_capacity(2 * rows);
        Self {
            x: axis(),
            y: axis(),
            z: dim.has_z().then(axis),
            m: dim.has_m().then(axis),
            offsets: Offsets::with_capacity(rows),
            failed: Vec::new(),
        }
    }

    fn push(&mut self, side: &Side, at: OnPart) {
        let value = |axis: &[f64]| interpolated(axis[at.start], axis[at.end], at.t);
        self.x.push(value(side.coords.x()));
        self.y.push(value(side.coords.y()));
        if let Some(z) = &mut self.z {
            z.push(value(
                side.coords
                    .z()
                    .expect("a z on both sides, with a z in the line"),
            ));
        }
        if let Some(m) = &mut self.m {
            m.push(value(
                side.coords
                    .m()
                    .expect("an m on both sides, with an m in the line"),
            ));
        }
    }

    fn end_row(&mut self) {
        let vertices = self.x.len() - *self.offsets.last() as usize;
        self.offsets
            .try_push(vertices)
            .expect("at most two vertices per row");
    }

    fn into_array(self, validity: Option<Bitmap>) -> LineStringArray {
        let rows = List::from_offsets(self.offsets.into(), validity).with_null_at(&self.failed);
        LineStringArray::from_parts(
            rows,
            ValidCoords::from_axes((self.x, self.y, self.z, self.m)),
        )
    }
}

/// Adds the line from row `i` of `a` to row `j` of `b`.
/// Pushes nothing if it fails.
fn push_line<const SKIP: bool>(
    lines: &mut Lines,
    a: &Side,
    i: usize,
    b: &Side,
    j: usize,
) -> GeoComputeResult<()> {
    if a.has_no_vertices(i) || b.has_no_vertices(j) {
        return Ok(());
    }
    // Before the search, as `SKIP` leaves out every coordinate that is not finite.
    a.ensure_row_passes(i)?;
    b.ensure_row_passes(j)?;
    // A side left without a vertex to search: the line is empty.
    let Some(a_to_b) = furthest::<SKIP>(a, i, b, j, f64::NEG_INFINITY) else {
        return Ok(());
    };
    // `b` to `a` only wins if it is strictly further, so that a tie goes to `a` to `b`.
    match furthest::<SKIP>(b, j, a, i, a_to_b.distance_squared) {
        Some(b_to_a) => {
            lines.push(a, b_to_a.nearest);
            lines.push(b, OnPart::vertex(b_to_a.vertex));
        }
        None => {
            lines.push(a, OnPart::vertex(a_to_b.vertex));
            lines.push(b, a_to_b.nearest);
        }
    }
    Ok(())
}

/// The lines from row `i` of `a` to row `j` of `b` for every `(i, j)` of `rows`.
/// A row that is null in `validity` gets a null line.
fn lines_between<const SKIP: bool>(
    a: &Side,
    b: &Side,
    rows: impl ExactSizeIterator<Item = (usize, usize)>,
    validity: Option<Bitmap>,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<LineStringArray> {
    let dim = a.coords.dim().shared_with(b.coords.dim());
    let mut lines = Lines::new(rows.len(), dim);
    for (row, (i, j)) in rows.enumerate() {
        if validity
            .as_ref()
            .is_none_or(|validity| validity.get_bit(row))
        {
            match push_line::<SKIP>(&mut lines, a, i, b, j) {
                Err(_) if ignore_per_row_errors => lines.failed.push(row),
                line => line?,
            }
        }
        lines.end_row();
    }
    Ok(lines.into_array(validity))
}

/// Gets the line spanning the Hausdorff distance from `a` to `b`, row by row.
/// See Python `hausdorff_line_between`.
/// Its axes are those both `a` and `b` have.
///
/// `a` and `b` must have the same length.
pub fn hausdorff_lines_pairwise(
    a: &GeometryArray,
    b: &GeometryArray,
    options: HausdorffOptions,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<LineStringArray> {
    debug_assert_eq!(a.len(), b.len(), "`a` and `b` must have the same length");
    let rows = (0..a.len()).map(|i| (i, i));
    let validity = combine_validities_and(a.validity(), b.validity());
    let (a, b) = (Side::new(a, options), Side::new(b, options));
    with_const_flags!(
        SKIP = options.skip();
        lines_between::<SKIP>(&a, &b, rows, validity, ignore_per_row_errors)
    )
}

/// Gets the line spanning the Hausdorff distance between a single geometry and every row of `many`,
/// as [hausdorff_lines_pairwise] does for a `one` repeated on every row.
/// With `lines_start_on_one`, every line runs from `one` to `many`, otherwise from `many` to `one`.
///
/// `one` must hold a single row.
pub fn hausdorff_lines_against_one(
    one: &GeometryArray,
    many: &GeometryArray,
    lines_start_on_one: bool,
    options: HausdorffOptions,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<LineStringArray> {
    assert_eq!(one.len(), 1, "`one` must be a single row");
    let len = many.len();
    let validity = match one.validity().is_none_or(|validity| validity.get_bit(0)) {
        true => many.validity().cloned(),
        false => Some(Bitmap::new_zeroed(len)),
    };
    let (one, many) = (Side::new(one, options), Side::new(many, options));
    let (a, b) = match lines_start_on_one {
        true => (&one, &many),
        false => (&many, &one),
    };
    let rows = (0..len).map(|i| match lines_start_on_one {
        true => (0, i),
        false => (i, 0),
    });
    with_const_flags!(
        SKIP = options.skip();
        lines_between::<SKIP>(a, b, rows, validity, ignore_per_row_errors)
    )
}
