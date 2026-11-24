//! The area a polygon encloses, as an [AreaMetric] measures it.
//! Implements the Python `area`.
//!
//! The const generics `EARLY_COORD_CHECK` and `EXTERIOR_SIDE` hold an [EarlyCoordCheck] and an [ExteriorSide].

use std::convert::Infallible;
use std::ops::Range;

use geographiclib_rs::{PolygonArea, Winding};
use geopolars_views::{
    List, MultiPolygonArray, PolygonArray, ScalarLineString, ScalarPolygon, Untrimmed, ValidCoords,
};
use polars_arrow::array::PrimitiveArray;
use polars_arrow::datatypes::ArrowDataType;

use super::{first_outside, has_none_outside, Bounds};
use crate::early_coord_check::every_xy_passes;
use crate::per_row::per_row_or_null;
use crate::proj::ReprojectedXy;
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

const_enum!(
    /// The `exterior_side` of the Python `area`.
    pub enum ExteriorSide {
        FitHoles,
        Smaller,
    }
);

/// How the area a ring encloses is measured
pub trait AreaMetric: BaseMetric {
    /// Why a row could not be measured (e.g. a failed reprojection).
    type Error;
    /// What is worked out once for a whole chunk, before its rings are measured.
    type Chunk;
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
    /// The area an exterior encloses, given its [`ring_area`](Self::ring_area)
    /// and the `holes` its polygon takes out of it.
    fn exterior_area<const EXTERIOR_SIDE: u8>(&self, exterior: f64, holes: f64) -> f64;
}

impl AreaMetric for Planar {
    type Error = Infallible;
    type Chunk = ();

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

    fn ring_area(&self, _chunk: &(), ring: &ScalarLineString) -> f64 {
        twice_signed_area(ring).abs() / 2.0
    }

    #[inline(always)]
    fn exterior_area<const EXTERIOR_SIDE: u8>(&self, exterior: f64, _holes: f64) -> f64 {
        // A ring on the plane has a single inside, so there is no side to pick.
        exterior
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

impl AreaMetric for GeodesicMetric {
    type Error = GeoComputeError;
    type Chunk = ReprojectedXy;

    fn prepare_chunk(&self, coords: &ValidCoords, vertices: Range<usize>) -> ReprojectedXy {
        self.lonlats(coords, vertices)
    }

    #[inline(always)]
    fn check_row(
        &self,
        lonlats: &ReprojectedXy,
        coords: &ValidCoords,
        vertices: Range<usize>,
    ) -> Result<(), GeoComputeError> {
        self.check_lonlats(lonlats, coords, vertices)
    }

    fn ring_area(&self, lonlats: &ReprojectedXy, ring: &ScalarLineString) -> f64 {
        // On the ellipsoid, in the CRS's unit squared, like [`between`](GeodesicMetric::between).
        // Based on `geographiclib_rs`.
        let mut area = PolygonArea::new(&self.ellipsoid, Winding::CounterClockwise);
        for i in ring.vertices.clone() {
            let (lon, lat) = lonlats.xy(i);
            // The repeated closing vertex adds an edge of length zero, which encloses nothing.
            area.add_point(lat, lon);
        }
        // Signed, then made positive.
        // unsigned would take a clockwise ring
        // to enclose everything outside of it.
        let (_perimeter, area, _vertices) = area.compute(true);
        area.abs()
    }

    #[inline(always)]
    fn exterior_area<const EXTERIOR_SIDE: u8>(&self, exterior: f64, holes: f64) -> f64 {
        // `ring_area` takes the smaller of the two sides of a ring.
        // Holes that take more than that only fit in the larger side, which is then the exterior's.
        // An exterior that encloses nothing (empty or degenerate) keeps its area of 0, rather than becoming the whole ellipsoid.
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
) -> f64 {
    let Some(exterior) = polygon.exterior() else {
        return 0.0;
    };
    let exterior = metric.ring_area(chunk, &exterior);
    let holes: f64 = polygon
        .holes()
        .map(|hole| metric.ring_area(chunk, &hole))
        .sum();
    let area = metric.exterior_area::<EXTERIOR_SIDE>(exterior, holes) - holes;
    // The Python `area` says which polygons come out below zero.
    if area < 0.0 && negative_area == NegativeArea::Zero {
        return 0.0;
    }
    area
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
    with_const_flags!(
        EXTERIOR_SIDE: ExteriorSide = exterior_side;
        areas_with(
            metric,
            array.rows(),
            array.coords(),
            |i| array.row_vertices(i),
            |chunk, i| {
                let polygon = array.get(i);
                polygon_area::<M, EXTERIOR_SIDE>(metric, chunk, &polygon, negative_area)
            },
            on_row_error,
            early_coord_check,
            bounds,
            on_out_of_bounds,
        )
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
    with_const_flags!(
        EXTERIOR_SIDE: ExteriorSide = exterior_side;
        areas_with(
            metric,
            array.rows(),
            array.coords(),
            |i| array.row_vertices(i),
            |chunk, i| {
                array
                    .get(i)
                    .polygons()
                    // Folded from `0.0`, as `sum` starts from `-0.0`.
                    .fold(0.0, |total, polygon| {
                        total
                            + polygon_area::<M, EXTERIOR_SIDE>(metric, chunk, &polygon, negative_area)
                    })
            },
            on_row_error,
            early_coord_check,
            bounds,
            on_out_of_bounds,
        )
    )
}

/// The area of every row of a chunk, as `row_area` measures a row that passes the checks.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn areas_with<M: AreaMetric<Error: Into<GeoComputeError>>>(
    metric: &M,
    rows: &List,
    coords: &ValidCoords,
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
    let chunk = metric.prepare_chunk(
        coords,
        areas_with_vertices_of_rows(rows.len(), &row_vertices),
    );
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
    if !areas_with_failed_check_row_passes_early::<EARLY_COORD_CHECK>(coords, vertices.clone()) {
        return Some(Ok(Some(f64::NAN)));
    }
    if !CHECK_BOUNDS {
        return None;
    }
    let (x, y) = (&coords.x()[vertices.clone()], &coords.y()[vertices]);
    if has_none_outside(bounds, x, y) {
        return None;
    }
    // `Ignore` never gets here, as it gives `CHECK_BOUNDS=false`.
    Some(match on_out_of_bounds {
        OnOutOfBounds::Null => Ok(None),
        _ => Err(GeoComputeError::OutOfBounds {
            position: first_outside(bounds, x, y)
                .expect("a coordinate outside, as the check above failed"),
            bounds: *bounds,
        }),
    })
}

#[inline(always)]
fn areas_with_failed_check_row_passes_early<const EARLY_COORD_CHECK: u8>(
    coords: &ValidCoords,
    vertices: Range<usize>,
) -> bool {
    if const { EarlyCoordCheck::from_u8(EARLY_COORD_CHECK) } == EarlyCoordCheck::Off {
        return true;
    }
    every_xy_passes::<EARLY_COORD_CHECK>(&coords.x()[vertices.clone()], &coords.y()[vertices])
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

fn areas_with_vertices_of_rows(
    row_count: usize,
    row_vertices: impl Fn(usize) -> Range<usize>,
) -> Range<usize> {
    if row_count == 0 {
        return 0..0;
    }
    // Offsets never decrease, also under a null row, so this spans the vertices of every row.
    row_vertices(0).start..row_vertices(row_count - 1).end
}
