//! The area a polygon encloses, as an [AreaMetric] measures it.
//! Implements the Python `area`.
//!
//! # Const generic flags
//!
//! ## `EARLY_COORD_CHECK`
//!
//! The kernels are generic over `EARLY_COORD_CHECK`, an [EarlyCoordCheck].
//! Unless it is `Off`, a row whose x or y fails it gets NaN, without being measured.
//!
//! ## `CHECK_BOUNDS`
//!
//! If `CHECK_BOUNDS=true`, a row with an x or y outside the bounds is not measured,
//! but handled as its [OnOutOfBounds] says.
//! It is `false` for [OnOutOfBounds::Ignore], or without bounds.

use std::convert::Infallible;
use std::ops::Range;

use geographiclib_rs::{PolygonArea, Winding};
use geopolars_views::{
    MultiPolygonArray, PolygonArray, ScalarLineString, ScalarPolygon, Untrimmed, ValidCoords,
};
use polars_arrow::array::PrimitiveArray;

use super::{first_outside, has_none_outside, Bounds};
use crate::early_coord_check::every_xy_passes;
use crate::per_row::per_row_or_null;
use crate::{
    BaseMetric, EarlyCoordCheck, GeoComputeError, GeoComputeResult, GeodesicMetric, OnOutOfBounds,
    OnRowError, Planar,
};

/// The `negative_area` of the Python `area`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeArea {
    Zero,
    Keep,
}

/// How the area a ring encloses is measured
pub trait AreaMetric: BaseMetric {
    /// Why a ring could not be measured (e.g. a failed reprojection).
    type Error;
    /// The area the closed `ring` encloses, whichever way it winds. Its heights do not count.
    fn ring_area(&self, ring: &ScalarLineString) -> Result<f64, Self::Error>;
}

impl AreaMetric for Planar {
    type Error = Infallible;

    fn ring_area(&self, ring: &ScalarLineString) -> Result<f64, Infallible> {
        Ok(twice_signed_area(ring).abs() / 2.0)
    }
}

/// How many partial sums [twice_signed_area] keeps, so that its adds do not wait on each other.
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

impl AreaMetric for GeodesicMetric {
    type Error = GeoComputeError;

    fn ring_area(&self, ring: &ScalarLineString) -> Result<f64, GeoComputeError> {
        // On the ellipsoid, in the CRS's unit squared, like [`between`](GeodesicMetric::between).
        // Based on `geographiclib_rs`.
        let mut area = PolygonArea::new(&self.ellipsoid, Winding::CounterClockwise);
        for [x, y, _] in ring.positions::<false>() {
            let (lon, lat) = self.lonlat((x, y))?;
            // The repeated closing vertex adds an edge of length zero, which encloses nothing.
            area.add_point(lat, lon);
        }
        // Signed, then made positive.
        // unsigned would take a clockwise ring
        // to enclose everything outside of it.
        let (_perimeter, area, _vertices) = area.compute(true);
        Ok(area.abs())
    }
}

#[inline(always)]
fn polygon_area<M: AreaMetric>(
    metric: &M,
    polygon: &ScalarPolygon,
    negative_area: NegativeArea,
) -> Result<f64, M::Error> {
    let Some(exterior) = polygon.exterior() else {
        return Ok(0.0);
    };
    let exterior = metric.ring_area(&exterior)?;
    let holes = polygon
        .holes()
        .map(|hole| metric.ring_area(&hole))
        .sum::<Result<f64, _>>()?;
    let area = exterior - holes;
    // Only a polygon that is not valid by OGC gets here:
    // an empty exterior with holes, or holes that take more than it encloses.
    if area < 0.0 && negative_area == NegativeArea::Zero {
        return Ok(0.0);
    }
    Ok(area)
}

#[inline(always)]
fn row_passes_early_check<const EARLY_COORD_CHECK: u8>(
    coords: &ValidCoords,
    vertices: Range<usize>,
) -> bool {
    if const { EarlyCoordCheck::from_u8(EARLY_COORD_CHECK) } == EarlyCoordCheck::Off {
        return true;
    }
    every_xy_passes::<EARLY_COORD_CHECK>(&coords.x()[vertices.clone()], &coords.y()[vertices])
}

/// What a row gets instead of its area, if it fails a check before being measured.
#[inline(always)]
fn instead_of_measuring<const EARLY_COORD_CHECK: u8, const CHECK_BOUNDS: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
    bounds: &Bounds,
    on_out_of_bounds: OnOutOfBounds,
) -> Option<GeoComputeResult<Option<f64>>> {
    if !row_passes_early_check::<EARLY_COORD_CHECK>(coords, vertices.clone()) {
        return Some(Ok(Some(f64::NAN)));
    }
    if !CHECK_BOUNDS {
        return None;
    }
    let (x, y) = (&coords.x()[vertices.clone()], &coords.y()[vertices]);
    if has_none_outside(bounds, x, y) {
        return None;
    }
    Some(match on_out_of_bounds {
        OnOutOfBounds::Null => Ok(None),
        _ => Err(GeoComputeError::OutOfBounds {
            position: first_outside(bounds, x, y)?,
            bounds: *bounds,
        }),
    })
}

/// Splits `bounds` into whether to check them and the bounds themselves,
/// dropping them for [OnOutOfBounds::Ignore].
/// The bounds are a placeholder when there is no check.
fn split_bounds_check(bounds: Option<Bounds>, on_out_of_bounds: OnOutOfBounds) -> (bool, Bounds) {
    let bounds = bounds.filter(|_| on_out_of_bounds != OnOutOfBounds::Ignore);
    (bounds.is_some(), bounds.unwrap_or_default())
}

/// The area of every polygon of a chunk.
pub fn polygon_areas<M: AreaMetric<Error: Into<GeoComputeError>>>(
    metric: &M,
    array: &Untrimmed<PolygonArray>,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
    bounds: Option<Bounds>,
    on_out_of_bounds: OnOutOfBounds,
    negative_area: NegativeArea,
) -> GeoComputeResult<PrimitiveArray<f64>> {
    let (check_bounds, bounds) = split_bounds_check(bounds, on_out_of_bounds);
    with_const_flags!(
        EARLY_COORD_CHECK: EarlyCoordCheck = early_coord_check, CHECK_BOUNDS = check_bounds;
        per_row_or_null(array.rows(), on_row_error, |i| {
            let vertices = array.row_vertices(i);
            if let Some(instead) = instead_of_measuring::<EARLY_COORD_CHECK, CHECK_BOUNDS>(
                array.coords(), vertices, &bounds, on_out_of_bounds,
            ) {
                return instead;
            }
            Ok(Some(polygon_area(metric, &array.get(i), negative_area).map_err(Into::into)?))
        })
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
    negative_area: NegativeArea,
) -> GeoComputeResult<PrimitiveArray<f64>> {
    let (check_bounds, bounds) = split_bounds_check(bounds, on_out_of_bounds);
    with_const_flags!(
        EARLY_COORD_CHECK: EarlyCoordCheck = early_coord_check, CHECK_BOUNDS = check_bounds;
        per_row_or_null(array.rows(), on_row_error, |i| {
            let vertices = array.row_vertices(i);
            if let Some(instead) = instead_of_measuring::<EARLY_COORD_CHECK, CHECK_BOUNDS>(
                array.coords(), vertices, &bounds, on_out_of_bounds,
            ) {
                return instead;
            }
            array
                .get(i)
                .polygons()
                // Folded from `0.0`, as `sum` starts from `-0.0`.
                .try_fold(0.0, |total, polygon| {
                    Ok(total + polygon_area(metric, &polygon, negative_area).map_err(Into::into)?)
                })
                .map(Some)
        })
    )
}
