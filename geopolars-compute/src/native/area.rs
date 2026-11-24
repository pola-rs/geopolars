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
//! A row that fails `EARLY_COORD_CHECK` gets its NaN first, even when it also lies outside them.
//! It is `false` for [OnOutOfBounds::Ignore], or without bounds.
//!
//! ## `EXTERIOR_SIDE`
//!
//! The kernels are generic over `EXTERIOR_SIDE`, an [ExteriorSide].
//! With `FitHoles`, a geodesic exterior whose holes take more than its smaller side gets its larger side,
//! the only one they fit in. With `Smaller`, every ring gets its smaller side.
//! Planar ignores it.

use std::convert::Infallible;
use std::ops::Range;

use geographiclib_rs::{PolygonArea, Winding};
use geopolars_views::{
    MultiPolygonArray, PolygonArray, ScalarLineString, ScalarPolygon, Untrimmed, ValidCoords,
};
use polars_arrow::array::PrimitiveArray;
use polars_arrow::datatypes::ArrowDataType;

use super::{first_outside, has_none_outside, Bounds};
use crate::early_coord_check::every_xy_passes;
use crate::per_row::per_row_or_null;
use crate::{
    BaseMetric, EarlyCoordCheck, GeoComputeError, GeoComputeResult, GeodesicMetric, LonLats,
    OnOutOfBounds, OnRowError, Planar,
};

/// The `negative_area` of the Python `area`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeArea {
    Zero,
    Keep,
}

const_enum!(
    /// The `exterior_side` of the Python `area`.
    pub enum ExteriorSide {
        FitHoles,
        Smaller,
    }
);

/// How the area a ring encloses is measured
pub trait AreaMetric: BaseMetric {
    /// Why a ring could not be measured (e.g. a failed reprojection).
    type Error;
    /// What is worked out once for a whole chunk, before its rings are measured.
    type Chunk;
    /// Prepares measuring the rings whose vertices lie within `vertices` of `coords`.
    fn prepare_chunk(&self, coords: &ValidCoords, vertices: Range<usize>) -> Self::Chunk;
    /// The area the closed `ring` encloses, whichever way it winds. Its heights do not count.
    ///
    /// `chunk` must come from [`prepare_chunk`](Self::prepare_chunk) over vertices including the ring's.
    fn ring_area(&self, chunk: &Self::Chunk, ring: &ScalarLineString) -> Result<f64, Self::Error>;
    /// The area an exterior encloses, given its [`ring_area`](Self::ring_area)
    /// and the `holes` its polygon takes out of it.
    fn exterior_area<const EXTERIOR_SIDE: u8>(&self, exterior: f64, holes: f64) -> f64;
}

impl AreaMetric for Planar {
    type Error = Infallible;
    type Chunk = ();

    fn prepare_chunk(&self, _coords: &ValidCoords, _vertices: Range<usize>) {}

    fn ring_area(&self, _chunk: &(), ring: &ScalarLineString) -> Result<f64, Infallible> {
        Ok(twice_signed_area(ring).abs() / 2.0)
    }

    /// A ring on the plane has one inside, so `EXTERIOR_SIDE` changes nothing.
    #[inline(always)]
    fn exterior_area<const EXTERIOR_SIDE: u8>(&self, exterior: f64, _holes: f64) -> f64 {
        exterior
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
    /// Every vertex is reprojected up front, in a single call to PROJ.
    type Chunk = LonLats;

    fn prepare_chunk(&self, coords: &ValidCoords, vertices: Range<usize>) -> LonLats {
        self.lonlats(coords, vertices)
    }

    fn ring_area(
        &self,
        lonlats: &LonLats,
        ring: &ScalarLineString,
    ) -> Result<f64, GeoComputeError> {
        // On the ellipsoid, in the CRS's unit squared, like [`between`](GeodesicMetric::between).
        // Based on `geographiclib_rs`.
        let mut area = PolygonArea::new(&self.ellipsoid, Winding::CounterClockwise);
        for i in ring.vertices.clone() {
            let (lon, lat) = self.lonlat_of(lonlats, ring.coords, i)?;
            // The repeated closing vertex adds an edge of length zero, which encloses nothing.
            area.add_point(lat, lon);
        }
        // Signed, then made positive.
        // unsigned would take a clockwise ring
        // to enclose everything outside of it.
        let (_perimeter, area, _vertices) = area.compute(true);
        Ok(area.abs())
    }

    #[inline(always)]
    fn exterior_area<const EXTERIOR_SIDE: u8>(&self, exterior: f64, holes: f64) -> f64 {
        // `ring_area` takes the smaller of the two sides of a ring.
        // Holes that take more than that only fit in the larger side, which is then the exterior's.
        // An exterior that encloses nothing (empty or degenerate) keeps it, rather than becoming the whole ellipsoid.
        let fit_holes = const { ExteriorSide::from_u8(EXTERIOR_SIDE) } == ExteriorSide::FitHoles;
        if fit_holes && 0.0 < exterior && exterior < holes {
            return self.ellipsoid.area() - exterior;
        }
        exterior
    }
}

#[inline(always)]
fn polygon_area<M: AreaMetric, const EXTERIOR_SIDE: u8>(
    metric: &M,
    chunk: &M::Chunk,
    polygon: &ScalarPolygon,
    negative_area: NegativeArea,
) -> Result<f64, M::Error> {
    let Some(exterior) = polygon.exterior() else {
        return Ok(0.0);
    };
    let exterior = metric.ring_area(chunk, &exterior)?;
    let holes = polygon
        .holes()
        .map(|hole| metric.ring_area(chunk, &hole))
        .sum::<Result<f64, _>>()?;
    let area = metric.exterior_area::<EXTERIOR_SIDE>(exterior, holes) - holes;
    // Only a polygon that is not valid by OGC gets here
    // (an empty exterior with holes, or holes that take more than it encloses),
    // or with `ExteriorSide::Smaller`, a valid one whose geodesic exterior covers over half the ellipsoid.
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

fn vertices_of_rows(
    row_count: usize,
    row_vertices: impl Fn(usize) -> Range<usize>,
) -> Range<usize> {
    if row_count == 0 {
        return 0..0;
    }
    row_vertices(0).start..row_vertices(row_count - 1).end
}

/// The area of every polygon of a chunk.
#[allow(clippy::too_many_arguments)]
pub fn polygon_areas<M: AreaMetric<Error: Into<GeoComputeError>>>(
    metric: &M,
    array: &Untrimmed<PolygonArray>,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
    bounds: Option<Bounds>,
    on_out_of_bounds: OnOutOfBounds,
    negative_area: NegativeArea,
    exterior_side: ExteriorSide,
) -> GeoComputeResult<PrimitiveArray<f64>> {
    if array.rows().is_all_null() {
        return Ok(PrimitiveArray::new_null(
            ArrowDataType::Float64,
            array.rows().len(),
        ));
    }
    let (check_bounds, bounds) = split_bounds_check(bounds, on_out_of_bounds);
    let vertices = vertices_of_rows(array.rows().len(), |i| array.row_vertices(i));
    let chunk = metric.prepare_chunk(array.coords(), vertices);
    with_const_flags!(
        EARLY_COORD_CHECK: EarlyCoordCheck = early_coord_check, CHECK_BOUNDS = check_bounds,
        EXTERIOR_SIDE: ExteriorSide = exterior_side;
        per_row_or_null(array.rows(), on_row_error, |i| {
            let vertices = array.row_vertices(i);
            if let Some(instead) = instead_of_measuring::<EARLY_COORD_CHECK, CHECK_BOUNDS>(
                array.coords(), vertices, &bounds, on_out_of_bounds,
            ) {
                return instead;
            }
            let polygon = array.get(i);
            Ok(Some(
                polygon_area::<M, EXTERIOR_SIDE>(metric, &chunk, &polygon, negative_area)
                    .map_err(Into::into)?,
            ))
        })
    )
}

/// The area of every multipolygon of a chunk.
#[allow(clippy::too_many_arguments)]
pub fn multi_polygon_areas<M: AreaMetric<Error: Into<GeoComputeError>>>(
    metric: &M,
    array: &Untrimmed<MultiPolygonArray>,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
    bounds: Option<Bounds>,
    on_out_of_bounds: OnOutOfBounds,
    negative_area: NegativeArea,
    exterior_side: ExteriorSide,
) -> GeoComputeResult<PrimitiveArray<f64>> {
    if array.rows().is_all_null() {
        return Ok(PrimitiveArray::new_null(
            ArrowDataType::Float64,
            array.rows().len(),
        ));
    }
    let (check_bounds, bounds) = split_bounds_check(bounds, on_out_of_bounds);
    let vertices = vertices_of_rows(array.rows().len(), |i| array.row_vertices(i));
    let chunk = metric.prepare_chunk(array.coords(), vertices);
    with_const_flags!(
        EARLY_COORD_CHECK: EarlyCoordCheck = early_coord_check, CHECK_BOUNDS = check_bounds,
        EXTERIOR_SIDE: ExteriorSide = exterior_side;
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
                    let area =
                        polygon_area::<M, EXTERIOR_SIDE>(metric, &chunk, &polygon, negative_area)
                            .map_err(Into::into)?;
                    Ok(total + area)
                })
                .map(Some)
        })
    )
}
