//! Pairing up the rows of two arrays of geometries, for a GEOS operation on two geometries.

use geopolars_core::output::SizeHint;
use geopolars_core::GeometryArray;
use polars_arrow::bitmap::Bitmap;

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
        let rows = pair_count(a.geometries.len(), b.geometries.len());
        Ok(Self { a, b, rows })
    }

    /// See [`GeosPairs::new`], for an operation whose result is empty when either row is.
    /// A row paired with an empty one (without parts) is therefore not converted,
    /// and is taken as empty too.
    /// A single row is only taken as empty when every row it is paired with is.
    pub(super) fn new_skipping_partners_of_empty(
        a: &GeometryArray,
        b: &GeometryArray,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<Self> {
        let (a_empty, b_empty) = (rows_without_parts(a), rows_without_parts(b));
        let rows = pair_count(a_empty.len(), b_empty.len());
        let (ignore, early) = (ignore_per_row_errors, early_finite_check);
        let a_as_empty = paired_with_empty(&b_empty, a_empty.len());
        let a = a.to_geos_rows_taking_as_empty(Some(&a_as_empty), ignore, early)?;
        let b_as_empty = paired_with_empty(&a_empty, b_empty.len());
        let b = b.to_geos_rows_taking_as_empty(Some(&b_as_empty), ignore, early)?;
        Ok(Self { a, b, rows })
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&GeosRow, &GeosRow)> {
        broadcast(&self.a.geometries, self.rows).zip(broadcast(&self.b.geometries, self.rows))
    }

    pub(super) fn len(&self) -> usize {
        self.rows
    }

    pub(super) fn sides(&self) -> (&[GeosRow], &[GeosRow]) {
        (&self.a.geometries, &self.b.geometries)
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

/// Marks the valid rows that have no parts.
/// A point has none when it is GeoArrow's empty point (NaN x and y).
fn rows_without_parts(array: &GeometryArray) -> Bitmap {
    let rows = match array {
        GeometryArray::Point(array) => {
            let coords = &array.coords;
            let is_empty = |i| coords.xy(i).is_some_and(|(x, y)| x.is_nan() && y.is_nan());
            return Bitmap::from_iter((0..coords.len()).map(is_empty));
        }
        GeometryArray::LineString(array) => &array.rows,
        GeometryArray::Polygon(array) => &array.rows,
        GeometryArray::MultiPoint(array) => &array.rows,
        GeometryArray::MultiLineString(array) => &array.rows,
        GeometryArray::MultiPolygon(array) => &array.rows,
    };
    // A null row's offsets point anywhere, so its validity is checked first.
    Bitmap::from_iter((0..rows.len()).map(|i| rows.is_valid(i) && rows.range(i).is_empty()))
}

/// Marks which of `len` rows are paired with an empty row of `partner_empty`, broadcast as in [`GeosPairs::iter`].
fn paired_with_empty(partner_empty: &Bitmap, len: usize) -> Bitmap {
    if partner_empty.len() == len {
        partner_empty.clone()
    } else if partner_empty.len() == 1 {
        Bitmap::new_with_value(partner_empty.get_bit(0), len)
    } else {
        // A single row, paired with every row of the partner.
        Bitmap::new_with_value(partner_empty.unset_bits() == 0, len)
    }
}

/// Gives `len` rows, repeating `rows` if it holds a single one.
///
/// `rows` must hold `len` rows, or a single one.
fn broadcast(rows: &[GeosRow], len: usize) -> impl Iterator<Item = &GeosRow> {
    rows.iter().cycle().take(len)
}
