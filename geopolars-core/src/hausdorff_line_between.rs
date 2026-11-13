use geopolars_compute::native::{hausdorff_lines_against_one, hausdorff_lines_pairwise, NonFinite};
use geopolars_views::{GeometryArray, Kind};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// Finds the line spanning the Hausdorff distance between the geometries on the same row
/// of a chunk of an `a_kind` column's storage and one of a `b_kind` column's.
/// Implements the Python `hausdorff_line_between`.
/// `a` and `b` must have the same length.
pub fn pairwise_hausdorff_lines(
    a: &dyn Array,
    a_kind: Kind,
    b: &dyn Array,
    b_kind: Kind,
    failing: NonFinite,
    ignore_per_row_errors: bool,
) -> PolarsResult<GeometryArray> {
    let (a, b) = (
        GeometryArray::new(a, a_kind)?,
        GeometryArray::new(b, b_kind)?,
    );
    hausdorff_lines_pairwise(&a, &b, failing, ignore_per_row_errors)
        .map(GeometryArray::LineString)
        .into_polars()
}

/// Finds the line spanning the Hausdorff distance between the geometry in `one`,
/// a single row of a `one_kind` column's storage,
/// and every geometry in a chunk of a `many_kind` column's storage.
/// Implements the Python `hausdorff_line_between`.
/// Every line starts on `one` if `lines_start_on_one=true`, and on the row of `many` otherwise.
pub fn hausdorff_lines_to_one(
    one: &dyn Array,
    one_kind: Kind,
    many: &dyn Array,
    many_kind: Kind,
    lines_start_on_one: bool,
    failing: NonFinite,
    ignore_per_row_errors: bool,
) -> PolarsResult<GeometryArray> {
    let one = GeometryArray::new(one, one_kind)?;
    let many = GeometryArray::new(many, many_kind)?;
    hausdorff_lines_against_one(
        &one,
        &many,
        lines_start_on_one,
        failing,
        ignore_per_row_errors,
    )
    .map(GeometryArray::LineString)
    .into_polars()
}
