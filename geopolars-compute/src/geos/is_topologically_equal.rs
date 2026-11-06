//! Whether two geometries are topologically equal, forwarded to GEOS.
//! See `is_topologically_equal` in Python for the rules.
//!
//! Both sides take any kind as a `&dyn ToGeosRows`,
//! so one kernel serves every pair of kinds.

use geos::Geom;
use polars_arrow::array::BooleanArray;

use super::{GeosRow, ToGeosRows};
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

/// Whether two arrays of geometries are topologically equal, row by row.
/// A row that GEOS fails to convert or compare makes this return its error.
/// That row is null instead if `ignore_per_row_errors=true`.
///
/// `a` and `b` must have the same length.
pub fn topologically_equal_pairwise(
    a: &dyn ToGeosRows,
    b: &dyn ToGeosRows,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<BooleanArray> {
    let a = a.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let b = b.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    debug_assert_eq!(
        a.geometries.len(),
        b.geometries.len(),
        "`a` and `b` must have the same length"
    );
    let tested = a
        .geometries
        .iter()
        .zip(&b.geometries)
        .map(|(a, b)| row_or_null(rows_are_topologically_equal(a, b), ignore_per_row_errors))
        .collect::<GeoComputeResult<Vec<_>>>()?;
    Ok(optional_bool_array(&tested))
}

/// Whether the single geometry `one` is topologically equal to each row of `many`.
/// Equality goes both ways, so this serves a single geometry on either side.
/// See [topologically_equal_pairwise] for the other arguments.
pub fn topologically_equal_against_one(
    one: &dyn ToGeosRows,
    many: &dyn ToGeosRows,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<BooleanArray> {
    let one = one.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let [one] = one.geometries.as_slice() else {
        panic!("`one` must be a single row, got {}", one.geometries.len());
    };
    let many = many.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let tested = many
        .geometries
        .iter()
        .map(|row| {
            row_or_null(
                rows_are_topologically_equal(one, row),
                ignore_per_row_errors,
            )
        })
        .collect::<GeoComputeResult<Vec<_>>>()?;
    Ok(optional_bool_array(&tested))
}
