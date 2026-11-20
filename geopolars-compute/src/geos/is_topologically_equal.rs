//! Whether two geometries are topologically equal, forwarded to GEOS.
//! Implements the Python `is_topologically_equal`.

use geopolars_views::GeometryArray;
use geos::{Geom, Geometry};
use polars_arrow::array::BooleanArray;

use super::pairs::{GeosPairs, PairOperation, WhenEmpty};
use crate::per_row::optional_bool_array;
use crate::{EarlyCoordCheck, GeoComputeResult, OnRowError};

/// See [`topologically_equal`].
/// Two empty rows are equal, and an empty row equals no row with vertices.
struct TopologicallyEqual;

impl PairOperation for TopologicallyEqual {
    type Output = bool;

    fn when_both_empty(&self) -> bool {
        true
    }

    fn when_a_empty(&self) -> WhenEmpty<Self> {
        WhenEmpty::Gives(|| false)
    }

    fn when_b_empty(&self) -> WhenEmpty<Self> {
        WhenEmpty::Gives(|| false)
    }

    fn on_geometries(&self, a: &Geometry, b: &Geometry) -> GeoComputeResult<bool> {
        Ok(a.equals(b)?)
    }
}

/// Tests whether two arrays of geometries are topologically equal, row by row.
/// Implements the Python `is_topologically_equal`.
/// A single row on either side is compared against every row of the other.
///
/// `a` and `b` must have the same length, or one of them a single row.
pub fn topologically_equal(
    a: &GeometryArray,
    b: &GeometryArray,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<BooleanArray> {
    let pairs = GeosPairs::new(a, b, &TopologicallyEqual, on_row_error, early_coord_check)?;
    let tested = pairs
        .evaluate(&TopologicallyEqual)
        .collect::<GeoComputeResult<Vec<_>>>()?;
    Ok(optional_bool_array(&tested))
}
