//! How far apart points are, as a [DistanceMetric] measures them,
//! either row by row ([pairwise]) or from one fixed point ([against_one]).
//! Implements the Python `distance`.
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! Every function here is generic over `Z`.
//! If `Z=true`, the difference in z counts towards the distance.
//!
//! ## `STOP_ON_ERROR`
//!
//! Every function here is generic over `STOP_ON_ERROR`.
//! If `STOP_ON_ERROR=true`, it stops at the first error and returns it.
//! If `STOP_ON_ERROR=false`, it marks that row null and carries on, so it never fails.
//! For [against_one], an error on `one` marks every row null.
//!
//! ## `NULLS`
//!
//! `pairwise_values` and `against_one_values` are generic over `NULLS`.
//! If `NULLS=true`, a null point is not measured.
//! If `NULLS=false`, every point is measured, null or not.

use std::convert::Infallible;

use geopolars_views::output::{f64_array, f64_array_masked};
use geopolars_views::PointArray;
use polars_arrow::array::PrimitiveArray;

use crate::{BaseMetric, GeoComputeError, GeodesicMetric, Planar, WithHeight};

/// How a segment is measured: on the plane, or along an ellipsoid.
/// Generic, so every metric and dimension gets a loop of its own.
pub trait DistanceMetric: BaseMetric {
    /// A vertex, ready to measure from (reprojected once, for the ellipsoid).
    type Vertex: Copy;
    /// Why a vertex could not be prepared (e.g. a failed reprojection).
    type Error;
    /// Sometimes it is faster to just *do* a calculation and then
    /// mask out the result *afterwards* (`TRUE`),
    /// rather than first checking for validity,
    /// and if the value is null,
    /// skipping the calculation all together (`FALSE`).
    /// This flag switches between the two behaviours.
    ///
    /// If `FALSE`,
    /// That slot can hold anything in its fields,
    /// so only a metric that measures anything harmlessly should!
    /// One that must never hand it on (e.g. to PROJ) skips null points.
    const MEASURES_NULL_SLOTS: bool;
    /// The vertex at `position`. Its height only counts in [`segment`](Self::segment) with `Z`.
    fn vertex(&self, position: [f64; 3]) -> Result<Self::Vertex, Self::Error>;
    /// How long the segment `ab` is.
    fn segment<const Z: bool>(&self, a: Self::Vertex, b: Self::Vertex) -> f64;
}

impl DistanceMetric for Planar {
    type Vertex = [f64; 3];
    type Error = Infallible;
    const MEASURES_NULL_SLOTS: bool = true;

    #[inline(always)]
    fn vertex(&self, position: [f64; 3]) -> Result<[f64; 3], Infallible> {
        Ok(position)
    }

    #[inline(always)]
    fn segment<const Z: bool>(&self, [ax, ay, az]: [f64; 3], [bx, by, bz]: [f64; 3]) -> f64 {
        let (dx, dy) = (bx - ax, by - ay);
        if Z {
            let dz = bz - az;
            (dx * dx + dy * dy + dz * dz).sqrt()
        } else {
            (dx * dx + dy * dy).sqrt()
        }
    }
}

/// Along the ellipsoid, climbing and descending with `z`:
/// each segment is its geodesic and its difference in height, by Pythagoras.
impl DistanceMetric for GeodesicMetric {
    type Vertex = WithHeight;
    type Error = GeoComputeError;
    const MEASURES_NULL_SLOTS: bool = false;

    #[inline(always)]
    fn vertex(&self, [x, y, z]: [f64; 3]) -> Result<WithHeight, GeoComputeError> {
        Ok((self.lonlat((x, y))?, z))
    }

    #[inline(always)]
    fn segment<const Z: bool>(&self, (a, z_a): WithHeight, (b, z_b): WithHeight) -> f64 {
        let along = self.between(a, b);
        if Z {
            let dz = z_b - z_a;
            (along * along + dz * dz).sqrt()
        } else {
            along
        }
    }
}

/// The distance between the points of `a` and `b`, which are the same length.
pub fn pairwise<const Z: bool, const STOP_ON_ERROR: bool, M: DistanceMetric>(
    metric: &M,
    a: &PointArray,
    b: &PointArray,
) -> Result<PrimitiveArray<f64>, M::Error> {
    debug_assert!(
        a.coords.len() == b.coords.len(),
        "Chunks must be the same size"
    );
    // Can't just pass the M::MEASURES_NULL_SLOTS into the function on rust stable,
    // so dispatch on it at runtime. I know we use nightly at polars,
    // but this is nicer if other people would want to use this crate :)
    // Without null points there is nothing to skip either.
    let measure_null_points_anyway =
        M::MEASURES_NULL_SLOTS || (a.coords.is_all_valid() && b.coords.is_all_valid());
    let (values, failed) = with_const_flags!(
        NULLS = !measure_null_points_anyway;
        pairwise_values::<NULLS, Z, STOP_ON_ERROR, _>(metric, a, b)
    )?;
    Ok(f64_array_masked(
        values,
        a.coords.validity(),
        b.coords.validity(),
        &failed,
    ))
}

/// [pairwise]'s distances, and the rows that failed.
/// A null point's distance is masked by the validity afterwards.
#[inline(always)]
fn pairwise_values<const NULLS: bool, const Z: bool, const STOP_ON_ERROR: bool, M>(
    metric: &M,
    a: &PointArray,
    b: &PointArray,
) -> Result<(Vec<f64>, Vec<usize>), M::Error>
where
    M: DistanceMetric,
{
    // Filled by index, not collected,
    // because collecting through `Result` loses the length,
    // so the Vec would grow as it goes, checking its capacity on every row.
    let mut values = vec![0.0; a.coords.len()];
    let mut failed = Vec::with_capacity(a.coords.len());
    for (i, value) in values.iter_mut().enumerate() {
        // An empty point (NaN coordinates) does go through, and comes out as a NaN distance:
        // empty points are assumed rare enough not to warrant an extra check on every valid point.
        if let (Some(here), Some(there)) = (
            a.get(i).position::<NULLS, Z>(),
            b.get(i).position::<NULLS, Z>(),
        ) {
            let distance = metric
                .vertex(here)
                .and_then(|here| Ok(metric.segment::<Z>(here, metric.vertex(there)?)));
            match distance {
                Ok(distance) => *value = distance,
                Err(e) if STOP_ON_ERROR => return Err(e),
                Err(_) => failed.push(i),
            }
        }
        // A null point keeps its 0.0, which the validity masks out.
    }
    Ok((values, failed))
}

/// The distance between the single point at `one` and every point of `many`.
/// `one` is prepared (reprojected) once, rather than on every row.
pub fn against_one<const Z: bool, const STOP_ON_ERROR: bool, M: DistanceMetric>(
    metric: &M,
    one: [f64; 3],
    many: &PointArray,
) -> Result<PrimitiveArray<f64>, M::Error> {
    // If any of the 'one' is NaN, then everything will become NaN anyway,
    // so we might as well exit early.
    // Null points stay null.
    if one[..if Z { 3 } else { 2 }]
        .iter()
        .any(|axis| axis.is_nan())
    {
        return Ok(f64_array(
            vec![f64::NAN; many.coords.len()],
            many.coords.validity(),
        ));
    }
    let one = match metric.vertex(one) {
        Ok(one) => one,
        Err(e) if STOP_ON_ERROR => return Err(e),
        Err(_) => {
            let every_row = (0..many.coords.len()).collect::<Vec<_>>();
            return Ok(f64_array_masked(
                vec![0.0; many.coords.len()],
                None,
                None,
                &every_row,
            ));
        }
    };
    let measure_null_points_anyway = M::MEASURES_NULL_SLOTS || many.coords.is_all_valid();
    let (values, failed) = with_const_flags!(
        NULLS = !measure_null_points_anyway;
        against_one_values::<NULLS, Z, STOP_ON_ERROR, _>(metric, one, many)
    )?;
    Ok(f64_array_masked(
        values,
        many.coords.validity(),
        None,
        &failed,
    ))
}

/// [against_one]'s distances, and the rows that failed.
/// A null point's distance is masked by the validity afterwards.
#[inline(always)]
fn against_one_values<const NULLS: bool, const Z: bool, const STOP_ON_ERROR: bool, M>(
    metric: &M,
    one: M::Vertex,
    many: &PointArray,
) -> Result<(Vec<f64>, Vec<usize>), M::Error>
where
    M: DistanceMetric,
{
    // Filled by index, as in `pairwise_values`.
    let mut values = vec![0.0; many.coords.len()];
    let mut failed = Vec::new();
    for (i, value) in values.iter_mut().enumerate() {
        // An empty point (NaN coordinates) does go through, and comes out as a NaN distance:
        // empty points are assumed rare enough not to warrant an extra check on every valid point.
        if let Some(there) = many.get(i).position::<NULLS, Z>() {
            match metric.vertex(there) {
                Ok(there) => *value = metric.segment::<Z>(one, there),
                Err(e) if STOP_ON_ERROR => return Err(e),
                Err(_) => failed.push(i),
            }
        }
        // A null point keeps its 0.0, which the validity masks out.
    }
    Ok((values, failed))
}
