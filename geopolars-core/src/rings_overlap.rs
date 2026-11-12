use geopolars_compute::geos::{rings_overlap_against_one, rings_overlap_pairwise};
use geopolars_compute::with_const_flags;
use geopolars_views::{downcast, LineStringArray};
use polars_arrow::array::{Array, BooleanArray};
use polars_core::prelude::{polars_err, PolarsError, PolarsResult};

fn compute_err(e: impl std::fmt::Display) -> PolarsError {
    polars_err!(ComputeError: "rings_overlap {e}")
}

fn rings(chunk: &dyn Array) -> PolarsResult<LineStringArray> {
    LineStringArray::new(downcast(chunk, "a list of vertices")?)
}

/// Checks whether the rings on the same row of two chunks of linestring storage overlap,
/// through GEOS.
/// Implements the Python `rings_overlap`.
pub fn pairwise_rings_overlap(
    a: &dyn Array,
    b: &dyn Array,
    include_touching: bool,
    ignore_per_row_errors: bool,
) -> PolarsResult<BooleanArray> {
    let (a, b) = (rings(a)?, rings(b)?);
    with_const_flags!(
        TOUCHING = include_touching;
        rings_overlap_pairwise::<TOUCHING>(&a, &b, ignore_per_row_errors)
    )
    .map_err(compute_err)
}

/// Checks whether the ring in `one`, a single row that is not null,
/// overlaps every ring in a chunk of linestring storage, through GEOS.
/// Implements the Python `rings_overlap`.
pub fn rings_overlap_one(
    one: &dyn Array,
    many: &dyn Array,
    include_touching: bool,
    ignore_per_row_errors: bool,
) -> PolarsResult<BooleanArray> {
    let (one, many) = (rings(one)?, rings(many)?);
    with_const_flags!(
        TOUCHING = include_touching;
        rings_overlap_against_one::<TOUCHING>(&one.get(0), &many, ignore_per_row_errors)
    )
    .map_err(compute_err)
}
