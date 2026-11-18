use geopolars_compute::geos::{shortest_lines_against_one, shortest_lines_pairwise};
use geopolars_compute::{EarlyCoordCheck, OnRowError};
use geopolars_views::{GeometryArray, Kind};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// Finds the shortest line between the geometries on the same row through GEOS,
/// for a chunk of an `a_kind` column's storage and one of a `b_kind` column's.
/// Implements the Python `shortest_line_between`.
///
/// `a` and `b` must have the same length.
pub fn pairwise_shortest_lines(
    a: &dyn Array,
    a_kind: Kind,
    b: &dyn Array,
    b_kind: Kind,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> PolarsResult<GeometryArray> {
    let (a, b) = (
        GeometryArray::new(a, a_kind)?,
        GeometryArray::new(b, b_kind)?,
    );
    shortest_lines_pairwise(a, b, on_row_error, early_coord_check)
        .map(GeometryArray::LineString)
        .into_polars()
}

/// Finds the shortest line between the geometry in `one`, of a `one_kind` column's storage,
/// and every geometry in a chunk of a `many_kind` column's storage, through GEOS.
/// Implements the Python `shortest_line_between`.
/// Every line starts on the row of `many`, unless `lines_start_on_one=true`.
///
/// `one` must be a single row.
pub fn shortest_lines_to_one(
    one: &dyn Array,
    one_kind: Kind,
    many: &dyn Array,
    many_kind: Kind,
    lines_start_on_one: bool,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> PolarsResult<GeometryArray> {
    let one = GeometryArray::new(one, one_kind)?;
    let many = GeometryArray::new(many, many_kind)?;
    shortest_lines_against_one(
        &one,
        &many,
        lines_start_on_one,
        on_row_error,
        early_coord_check,
    )
    .map(GeometryArray::LineString)
    .into_polars()
}
