use geopolars_compute::geos;
use geopolars_compute::{EarlyCoordCheck, OnRowError};
use geopolars_views::{GeometryArray, Kind};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// Finds the shortest line between the geometries on the same row through GEOS,
/// for a chunk of an `a_kind` column's storage and one of a `b_kind` column's.
/// Implements the Python `shortest_line_between`.
/// A single row on either side is paired with every row of the other.
///
/// `a` and `b` must have the same length, or one of them a single row.
pub fn shortest_lines(
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
    geos::shortest_lines(&a, &b, on_row_error, early_coord_check)
        .map(GeometryArray::LineString)
        .into_polars()
}
