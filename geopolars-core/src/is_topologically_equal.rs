use geopolars_compute::geos::topologically_equal;
use geopolars_views::{GeometryArray, Kind};
use polars_arrow::array::{Array, BooleanArray};
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// Checks whether the geometries on the same row of a chunk of an `a_kind` column's storage
/// and one of a `b_kind` column's are topologically equal, through GEOS.
/// Implements the Python `is_topologically_equal`.
/// See [`topologically_equal`] for the lengths `a` and `b` may have.
pub fn are_topologically_equal(
    a: &dyn Array,
    a_kind: Kind,
    b: &dyn Array,
    b_kind: Kind,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> PolarsResult<BooleanArray> {
    let (a, b) = (
        GeometryArray::new(a, a_kind)?,
        GeometryArray::new(b, b_kind)?,
    );
    topologically_equal(&a, &b, ignore_per_row_errors, early_finite_check).into_polars()
}
