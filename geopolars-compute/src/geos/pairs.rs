//! Pairing up the rows of two arrays of geometries, for a GEOS operation on two geometries.

use geopolars_core::output::SizeHint;
use geopolars_core::GeometryArray;
use polars_arrow::bitmap::Bitmap;

use super::{rows_without_vertices, GeosGeometries, GeosRow, ToGeosRows};
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
        let rows = pair_count(a.geometries.len(), b.geometries.len());
        Ok(Self { a, b, rows })
    }

    /// See [`GeosPairs::new`], for an operation whose result is empty when either row is empty,
    /// and null when either row is null.
    /// A row paired with an empty one (see [`rows_without_vertices`]) or a null is therefore not converted,
    /// and is taken as empty.
    /// A single row is only taken as empty when every row it is paired with is empty or null.
    pub(super) fn new_skipping_partners_of_empty(
        a: &GeometryArray,
        b: &GeometryArray,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<Self> {
        let a_decides = &rows_without_vertices(a) | &null_rows(a);
        let b_decides = &rows_without_vertices(b) | &null_rows(b);
        let a_as_empty = paired_with(&b_decides, a.len());
        let b_as_empty = paired_with(&a_decides, b.len());
        let (ignore, early) = (ignore_per_row_errors, early_finite_check);
        Self::new_taking_as_empty(a, b, &a_as_empty, &b_as_empty, ignore, early)
    }

    /// See [`GeosPairs::new`], for an operation whose result is null when either row is null,
    /// and empty when `a` is empty, but not when `b` is.
    /// A row paired with a null is therefore not converted, and is taken as empty.
    /// The same holds for a row of `b` paired with an empty `a`.
    /// Every row without vertices (see [`rows_without_vertices`]) is `Empty` as well.
    /// A single row is only taken as empty when this holds for every row it is paired with.
    pub(super) fn new_skipping_partners_of_empty_a(
        a: &GeometryArray,
        b: &GeometryArray,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<Self> {
        let (a_empty, b_empty) = (rows_without_vertices(a), rows_without_vertices(b));
        let a_as_empty = &a_empty | &paired_with(&null_rows(b), a.len());
        let a_decides = &a_empty | &null_rows(a);
        let b_as_empty = &b_empty | &paired_with(&a_decides, b.len());
        let (ignore, early) = (ignore_per_row_errors, early_finite_check);
        Self::new_taking_as_empty(a, b, &a_as_empty, &b_as_empty, ignore, early)
    }

    fn new_taking_as_empty(
        a: &GeometryArray,
        b: &GeometryArray,
        a_as_empty: &Bitmap,
        b_as_empty: &Bitmap,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<Self> {
        let rows = pair_count(a_as_empty.len(), b_as_empty.len());
        let (ignore, early) = (ignore_per_row_errors, early_finite_check);
        let a = a.to_geos_rows_taking_as_empty(Some(a_as_empty), ignore, early)?;
        let b = b.to_geos_rows_taking_as_empty(Some(b_as_empty), ignore, early)?;
        Ok(Self { a, b, rows })
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&GeosRow, &GeosRow)> {
        broadcast(&self.a.geometries, self.rows).zip(broadcast(&self.b.geometries, self.rows))
    }

    pub(super) fn len(&self) -> usize {
        self.rows
    }

    pub(super) fn size_hints(&self) -> (SizeHint<2>, SizeHint<2>) {
        (self.a.size_hint, self.b.size_hint)
    }
}

fn pair_count(a_rows: usize, b_rows: usize) -> usize {
    let rows = if a_rows == 1 { b_rows } else { a_rows };
    assert!(
        b_rows == rows || b_rows == 1,
        "cannot pair {a_rows} geometries with {b_rows}"
    );
    rows
}

fn null_rows(array: &GeometryArray) -> Bitmap {
    match array.validity() {
        Some(validity) => !validity,
        None => Bitmap::new_zeroed(array.len()),
    }
}

/// Marks which of `len` rows are paired with a marked row of `partner_marked`, broadcast as in [`GeosPairs::iter`].
/// A single row is only marked when every row of the partner is.
fn paired_with(partner_marked: &Bitmap, len: usize) -> Bitmap {
    if partner_marked.len() == len {
        partner_marked.clone()
    } else if partner_marked.len() == 1 {
        Bitmap::new_with_value(partner_marked.get_bit(0), len)
    } else {
        // A single row, paired with every row of the partner.
        Bitmap::new_with_value(partner_marked.unset_bits() == 0, len)
    }
}

/// Gives `len` rows, repeating `rows` if it holds a single one.
///
/// `rows` must hold `len` rows, or a single one.
fn broadcast(rows: &[GeosRow], len: usize) -> impl Iterator<Item = &GeosRow> {
    rows.iter().cycle().take(len)
}
