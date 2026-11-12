use geopolars_compute::geos::{shortest_lines_against_one, shortest_lines_pairwise};
use geopolars_views::{GeometryArray, Kind, LineStringArray};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// Finds the shortest line between the geometries on the same row of a chunk of an `a_kind` column's storage
/// and one of a `b_kind` column's, through GEOS.
/// Implements the Python `shortest_line_between`.
/// `a` and `b` must have the same length.
pub fn pairwise_shortest_lines(
    a: &dyn Array,
    a_kind: Kind,
    b: &dyn Array,
    b_kind: Kind,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> PolarsResult<LineStringArray> {
    let (a, b) = (
        GeometryArray::new(a, a_kind)?,
        GeometryArray::new(b, b_kind)?,
    );
    shortest_lines_pairwise(a, b, ignore_per_row_errors, early_finite_check).into_polars()
}

/// Finds the shortest line between the geometry in `one`, a single row of a `one_kind` column's storage,
/// and every geometry in a chunk of a `many_kind` column's storage, through GEOS.
/// Implements the Python `shortest_line_between`.
/// Every line starts on `one` if `lines_start_on_one=true`, and on the row of `many` otherwise.
pub fn shortest_lines_to_one(
    one: &dyn Array,
    one_kind: Kind,
    many: &dyn Array,
    many_kind: Kind,
    lines_start_on_one: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> PolarsResult<LineStringArray> {
    let one = GeometryArray::new(one, one_kind)?;
    let many = GeometryArray::new(many, many_kind)?;
    shortest_lines_against_one(
        &one,
        &many,
        lines_start_on_one,
        ignore_per_row_errors,
        early_finite_check,
    )
    .into_polars()
}
