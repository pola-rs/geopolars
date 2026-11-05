//! Whether two geometries are topologically equal, forwarded to GEOS.
//! See `is_equal` in Python for the rules.
//!
//! The geometries come in already converted (see [`ToGeosRows`](super::ToGeosRows)),
//! so one kernel serves every pair of kinds.

use geos::Geom;
use polars_arrow::array::BooleanArray;

use super::{GeosGeometries, GeosRow};
use crate::per_row::optional_bool_array;
use crate::{row_or_null, GeoComputeResult};

/// `None` if either is null.
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

/// Whether the rows of `a` and `b`, which have the same length, are topologically equal row by row.
/// A row is null where either geometry is.
/// If `ignore_per_row_errors=true`, a row GEOS fails on is null too;
/// otherwise its error is returned.
pub fn topologically_equal_pairwise(
    a: &GeosGeometries,
    b: &GeosGeometries,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<BooleanArray> {
    debug_assert_eq!(
        a.geometries.len(),
        b.geometries.len(),
        "Chunks must be the same size"
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
/// Complementary to [topologically_equal_pairwise]. See it for the other options.
/// Equality goes both ways, so this serves a single geometry on either side.
pub fn topologically_equal_against_one(
    one: &GeosRow,
    many: &GeosGeometries,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<BooleanArray> {
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
