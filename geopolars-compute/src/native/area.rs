//! The area a polygon encloses, as an [AreaMetric] measures it.
//! Implements the Python `area`.
//!
//! The const generic `EARLY_COORD_CHECK` holds an [EarlyCoordCheck].

use std::convert::Infallible;
use std::ops::Range;

use geopolars_views::{
    List, MultiPolygonArray, PolygonArray, ScalarLineString, ScalarPolygon, Untrimmed, ValidCoords,
};
use polars_arrow::array::PrimitiveArray;
use polars_arrow::datatypes::ArrowDataType;

use super::coordinate_selection::first_outside;
use super::Bounds;
use crate::early_coord_check::every_xy_passes_at;
use crate::per_row::per_row_or_null;
use crate::{
    BaseMetric, EarlyCoordCheck, GeoComputeError, GeoComputeResult, OnOutOfBounds, OnRowError,
    Planar,
};

/// How the area a ring encloses is measured
pub trait AreaMetric: BaseMetric {
    /// Why a row could not be measured (e.g. a failed reprojection).
    type Error;
    /// What is worked out once for a whole chunk, before its rings are measured.
    type Chunk;
    /// Whether the kernel runs `early_coord_check` before measuring a row.
    /// `false` where measuring costs no more than the check does.
    const CHECKS_COORDS_EARLY: bool;
    /// Prepares measuring the rings whose vertices lie within `vertices` of `coords`.
    fn prepare_chunk(&self, coords: &ValidCoords, vertices: Range<usize>) -> Self::Chunk;
    /// Fails a row that cannot be measured, before its rings are.
    /// `chunk` must come from [`prepare_chunk`](Self::prepare_chunk) over `coords`.
    fn check_row(
        &self,
        chunk: &Self::Chunk,
        coords: &ValidCoords,
        vertices: Range<usize>,
    ) -> Result<(), Self::Error>;
    /// The area the closed `ring` encloses, whichever way it winds. Its heights do not count.
    ///
    /// The ring's row must have passed [`check_row`](Self::check_row).
    fn ring_area(&self, chunk: &Self::Chunk, ring: &ScalarLineString) -> f64;
}

impl AreaMetric for Planar {
    type Error = Infallible;
    type Chunk = ();
    const CHECKS_COORDS_EARLY: bool = false;

    fn prepare_chunk(&self, _coords: &ValidCoords, _vertices: Range<usize>) {}

    #[inline(always)]
    fn check_row(
        &self,
        _chunk: &(),
        _coords: &ValidCoords,
        _vertices: Range<usize>,
    ) -> Result<(), Infallible> {
        Ok(())
    }

    #[inline(always)]
    fn ring_area(&self, _chunk: &(), ring: &ScalarLineString) -> f64 {
        twice_signed_area(ring).abs() / 2.0
    }
}

const PARTIAL_SUMS: usize = 8;

/// Twice the area the closed `ring` encloses in x and y, positive when it runs counter-clockwise.
#[inline(always)]
pub(super) fn twice_signed_area(ring: &ScalarLineString) -> f64 {
    // By the shoelace formula, as the sum of x_i * (y_{i+1} - y_{i-1}) over the vertices.
    // x is taken relative to the first vertex, so coordinates far from the origin keep their precision.
    // y only appears in differences, which already cancel its offset.
    let x = &ring.coords.x()[ring.vertices.clone()];
    let y = &ring.coords.y()[ring.vertices.clone()];
    let n = x.len();
    if n < 3 {
        return 0.0;
    }
    let x0 = x[0];
    let term = |x: f64, y_previous: f64, y_next: f64| (x - x0) * (y_next - y_previous);

    // The first vertex adds nothing, as its x is 0.
    let (x_chunks, x_rest) = x[1..n - 1].as_chunks::<PARTIAL_SUMS>();
    let (previous_chunks, previous_rest) = y[..n - 2].as_chunks::<PARTIAL_SUMS>();
    let (next_chunks, next_rest) = y[2..].as_chunks::<PARTIAL_SUMS>();
    // Several partial sums, so that the adds do not wait on each other.
    let mut sums = [0.0; PARTIAL_SUMS];
    for ((x, previous), next) in x_chunks.iter().zip(previous_chunks).zip(next_chunks) {
        for lane in 0..PARTIAL_SUMS {
            sums[lane] += term(x[lane], previous[lane], next[lane]);
        }
    }
    let mut twice_signed: f64 = sums.iter().sum();
    for ((&x, &previous), &next) in x_rest.iter().zip(previous_rest).zip(next_rest) {
        twice_signed += term(x, previous, next);
    }
    // The last vertex wraps around to the first.
    // That adds nothing to a closed ring, but keeps an unclosed one right.
    twice_signed + term(x[n - 1], y[n - 2], y[0])
}

#[inline(always)]
fn polygon_area<M: AreaMetric>(metric: &M, chunk: &M::Chunk, polygon: &ScalarPolygon) -> f64 {
    // A closed ring of fewer than 4 vertices can at most go there and back.
    // It gets 0.0 even with a NaN or infinite coordinate, as being too short counts for more.
    let ring_area = |ring: &ScalarLineString| {
        if ring.vertices.len() < 4 {
            0.0
        } else {
            metric.ring_area(chunk, ring)
        }
    };
    let Some(exterior) = polygon.exterior() else {
        return 0.0;
    };
    let exterior = ring_area(&exterior);
    let holes: f64 = polygon.holes().map(|hole| ring_area(&hole)).sum();
    exterior - holes
}

/// The area of every polygon of a chunk.
pub fn polygon_areas<M: AreaMetric<Error: Into<GeoComputeError>>>(
    metric: &M,
    array: &Untrimmed<PolygonArray>,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
    bounds: Option<Bounds>,
    on_out_of_bounds: OnOutOfBounds,
) -> GeoComputeResult<PrimitiveArray<f64>> {
    let rows = array.rows();
    let chunk_vertices = array.rings().span(rows.span(0..rows.len()));
    areas_with(
        metric,
        rows,
        array.coords(),
        chunk_vertices,
        |i| array.row_vertices(i),
        |chunk, i| {
            let polygon = array.get(i);
            polygon_area(metric, chunk, &polygon)
        },
        on_row_error,
        early_coord_check,
        bounds,
        on_out_of_bounds,
    )
}

/// The area of every multipolygon of a chunk.
pub fn multi_polygon_areas<M: AreaMetric<Error: Into<GeoComputeError>>>(
    metric: &M,
    array: &Untrimmed<MultiPolygonArray>,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
    bounds: Option<Bounds>,
    on_out_of_bounds: OnOutOfBounds,
) -> GeoComputeResult<PrimitiveArray<f64>> {
    let rows = array.rows();
    let chunk_vertices = array
        .rings()
        .span(array.polygons().span(rows.span(0..rows.len())));
    areas_with(
        metric,
        rows,
        array.coords(),
        chunk_vertices,
        |i| array.row_vertices(i),
        |chunk, i| {
            array
                .get(i)
                .polygons()
                // Folded from `0.0`, as `sum` starts from `-0.0`.
                .fold(0.0, |total, polygon| {
                    total + polygon_area(metric, chunk, &polygon)
                })
        },
        on_row_error,
        early_coord_check,
        bounds,
        on_out_of_bounds,
    )
}

/// The area of every row of a chunk, as `row_area` measures a row that passes the checks.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn areas_with<M: AreaMetric<Error: Into<GeoComputeError>>>(
    metric: &M,
    rows: &List,
    coords: &ValidCoords,
    chunk_vertices: Range<usize>,
    row_vertices: impl Fn(usize) -> Range<usize>,
    row_area: impl Fn(&M::Chunk, usize) -> f64,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
    bounds: Option<Bounds>,
    on_out_of_bounds: OnOutOfBounds,
) -> GeoComputeResult<PrimitiveArray<f64>> {
    if rows.is_all_null() {
        return Ok(PrimitiveArray::new_null(ArrowDataType::Float64, rows.len()));
    }
    let (check_bounds, bounds) = areas_with_split_bounds(bounds, on_out_of_bounds);
    let chunk = metric.prepare_chunk(coords, chunk_vertices);
    // A metric that does not check gets the `Off` path, whatever the user passed.
    let early_coord_check = if M::CHECKS_COORDS_EARLY {
        early_coord_check
    } else {
        EarlyCoordCheck::Off
    };
    with_const_flags!(
        EARLY_COORD_CHECK: EarlyCoordCheck = early_coord_check, CHECK_BOUNDS = check_bounds;
        per_row_or_null(rows, on_row_error, |i| {
            let vertices = row_vertices(i);
            if let Some(instead) = areas_with_failed_check::<EARLY_COORD_CHECK, CHECK_BOUNDS>(
                coords, vertices.clone(), &bounds, on_out_of_bounds,
            ) {
                return instead;
            }
            metric.check_row(&chunk, coords, vertices).map_err(Into::into)?;
            Ok(Some(row_area(&chunk, i)))
        })
    )
}

/// What a row gets instead of its area, if it fails a check before being measured.
#[inline(always)]
fn areas_with_failed_check<const EARLY_COORD_CHECK: u8, const CHECK_BOUNDS: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
    bounds: &Bounds,
    on_out_of_bounds: OnOutOfBounds,
) -> Option<GeoComputeResult<Option<f64>>> {
    if !every_xy_passes_at::<EARLY_COORD_CHECK>(coords.x(), coords.y(), vertices.clone()) {
        return Some(Ok(Some(f64::NAN)));
    }
    if !CHECK_BOUNDS {
        return None;
    }
    let (x, y) = (&coords.x()[vertices.clone()], &coords.y()[vertices]);
    let position = first_outside(bounds, x, y)?;
    // `Ignore` never gets here, as it gives `CHECK_BOUNDS=false`.
    Some(match on_out_of_bounds {
        OnOutOfBounds::Null => Ok(None),
        _ => Err(GeoComputeError::OutOfBounds {
            position,
            bounds: *bounds,
        }),
    })
}

/// Splits `bounds` into whether to check them and the bounds themselves,
/// dropping them for [OnOutOfBounds::Ignore].
/// The bounds are a placeholder when there is no check.
fn areas_with_split_bounds(
    bounds: Option<Bounds>,
    on_out_of_bounds: OnOutOfBounds,
) -> (bool, Bounds) {
    let bounds = bounds.filter(|_| on_out_of_bounds != OnOutOfBounds::Ignore);
    (bounds.is_some(), bounds.unwrap_or_default())
}
