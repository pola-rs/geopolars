use geopolars_compute::geos::RelatePairs;
use geopolars_views::{GeometryArray, Kind};
use polars_arrow::array::{Array, Utf8ViewArray};
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// Finds the DE-9IM matrix of the geometries on the same row of a chunk of an `a_kind` column's storage
/// and one of a `b_kind` column's, through GEOS.
/// Implements the Python `relate`.
/// See [`RelatePairs::new`] for the lengths `a` and `b` may have.
pub fn relate(
    a: &dyn Array,
    a_kind: Kind,
    b: &dyn Array,
    b_kind: Kind,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> PolarsResult<Utf8ViewArray> {
    let (a, b) = (
        GeometryArray::new(a, a_kind)?,
        GeometryArray::new(b, b_kind)?,
    );
    RelatePairs::new(&a, &b, ignore_per_row_errors, early_finite_check)
        .and_then(|pairs| pairs.to_strings())
        .into_polars()
}
