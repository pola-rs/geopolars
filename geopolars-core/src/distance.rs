use std::fmt::Display;

use geopolars_compute::native::{against_one, pairwise, DistanceMetric};
use geopolars_compute::with_const_flags;
use geopolars_views::{Coords, PointArray};
use polars_arrow::array::{Array, PrimitiveArray};
use polars_core::prelude::{polars_err, PolarsResult};

fn points(chunk: &dyn Array) -> PolarsResult<PointArray> {
    Ok(PointArray::new(Coords::new(chunk)?))
}

fn both_have_z(a: &PointArray, b: &PointArray) -> bool {
    a.coords.dim().has_z() && b.coords.dim().has_z()
}

/// Measures the distance between the points on the same row of two chunks of point storage,
/// as `metric` measures it.
/// Implements the Python `distance`.
///
/// `a` and `b` must have the same length.
pub fn pairwise_distances(
    a: &dyn Array,
    b: &dyn Array,
    metric: &impl DistanceMetric<Error: Display>,
) -> PolarsResult<PrimitiveArray<f64>> {
    let (a, b) = (points(a)?, points(b)?);
    with_const_flags!(Z = both_have_z(&a, &b); pairwise::<Z, true, _>(metric, &a, &b))
        .map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Measures the distance from the point in `one`
/// to every point in a chunk of point storage, as `metric` measures it.
/// Implements the Python `distance`.
///
/// `one` must be a single row that is not null.
pub fn distances_to_one(
    one: &dyn Array,
    many: &dyn Array,
    metric: &impl DistanceMetric<Error: Display>,
) -> PolarsResult<PrimitiveArray<f64>> {
    let (one, many) = (points(one)?, points(many)?);
    with_const_flags!(Z = both_have_z(&one, &many); {
        let one = one
            .get(0)
            .position::<false, Z>()
            .expect("`NULLS=false` always gives one");
        against_one::<Z, true, _>(metric, one, &many)
    })
    .map_err(|e| polars_err!(ComputeError: "{e}"))
}
