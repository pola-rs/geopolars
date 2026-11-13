//! Whether two geometries are topologically equal, forwarded to GEOS.
//! Implements the Python `is_topologically_equal`.

use geopolars_views::GeometryArray;
use geos::Geom;
use polars_arrow::array::BooleanArray;

use super::pairs::GeosPairs;
use super::GeosRow;
use crate::per_row::optional_bool_array;
use crate::{row_or_null, GeoComputeResult};

#[inline(always)]
fn rows_are_topologically_equal(a: &GeosRow, b: &GeosRow) -> GeoComputeResult<Option<bool>> {
    Ok(Some(match (a, b) {
        (GeosRow::Null, _) | (_, GeosRow::Null) => return Ok(None),
        (GeosRow::Empty, GeosRow::Empty) => true,
        // An empty row never went to GEOS, so it can't be compared there.
        (GeosRow::Empty, GeosRow::Geometry(other)) | (GeosRow::Geometry(other), GeosRow::Empty) => {
            other.is_empty()?
        }
        (GeosRow::Geometry(a), GeosRow::Geometry(b)) => a.equals(b)?,
    }))
}

/// Tests whether two arrays of geometries are topologically equal, row by row.
/// Implements the Python `is_topologically_equal`.
/// A single row on either side is compared against every row of the other.
/// Each side is converted to GEOS in full, so a row paired with a null can still fail.
///
/// `a` and `b` must have the same length, or one of them a single row.
pub fn topologically_equal(
    a: &GeometryArray,
    b: &GeometryArray,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<BooleanArray> {
    // TODO: Skip the partner of an empty row, as `intersection` does,
    // with emptiness by vertices as `rows_without_vertices` gives it:
    // `MULTIPOLYGON (EMPTY)` has a part, and GEOS finds it equal to `POLYGON EMPTY`.
    let pairs = GeosPairs::new(a, b, ignore_per_row_errors, early_finite_check)?;
    let tested = pairs
        .iter()
        .map(|(a, b)| row_or_null(rows_are_topologically_equal(a, b), ignore_per_row_errors))
        .collect::<GeoComputeResult<Vec<_>>>()?;
    Ok(optional_bool_array(&tested))
}
