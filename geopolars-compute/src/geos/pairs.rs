//! Pairing up the rows of two arrays of geometries, for a GEOS operation on two geometries.

use geopolars_core::output::SizeHint;
use geopolars_core::GeometryArray;

use super::{GeosGeometries, GeosRow, ToGeosRows};
use crate::GeoComputeResult;

/// Two arrays of geometries converted to GEOS, with their rows paired up.
/// A single row on either side is paired with every row of the other,
/// and stays on its own side, for an operation that is not symmetric.
pub(super) struct GeosPairs {
    // `b` is declared first so it drops first, freeing the geometries in reverse of building them.
    // With glibc's malloc, that makes `is_topologically_equal` on 50k polygons ~20% faster.
    b: GeosGeometries,
    a: GeosGeometries,
    rows: usize,
}

impl GeosPairs {
    /// Each side is converted to GEOS in full, so a row paired with a null can still fail.
    ///
    /// `a` and `b` must have the same length, or one of them a single row.
    pub(super) fn new(
        a: &GeometryArray,
        b: &GeometryArray,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<Self> {
        let a = a.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
        let b = b.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
        let (a_rows, b_rows) = (a.geometries.len(), b.geometries.len());
        let rows = if a_rows == 1 { b_rows } else { a_rows };
        assert!(
            b_rows == rows || b_rows == 1,
            "cannot pair {a_rows} geometries with {b_rows}"
        );
        Ok(Self { a, b, rows })
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&GeosRow, &GeosRow)> {
        broadcast(&self.a.geometries, self.rows).zip(broadcast(&self.b.geometries, self.rows))
    }

    pub(super) fn size_hints(&self) -> (SizeHint<2>, SizeHint<2>) {
        (self.a.size_hint, self.b.size_hint)
    }
}

/// `len` rows out of `rows`, which holds that many or a single one to repeat.
fn broadcast(rows: &[GeosRow], len: usize) -> impl Iterator<Item = &GeosRow> {
    rows.iter().cycle().take(len)
}
