//! Pairing up the rows of two arrays of geometries, for a GEOS operation on two geometries.

use std::ops::Range;

use geopolars_core::output::SizeHint;
use geopolars_core::{GeometryArray, List};
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
    /// A row paired with an empty one (see [`rows_without_vertices`]) is therefore not converted,
    /// and is taken as empty too.
    /// A single row is only taken as empty when every row it is paired with is.
    pub(super) fn new_skipping_partners_of_empty(
        a: &GeometryArray,
        b: &GeometryArray,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<Self> {
        let (a_empty, b_empty) = (rows_without_vertices(a), rows_without_vertices(b));
        let a_as_empty = paired_with_empty(&b_empty, a_empty.len());
        let b_as_empty = paired_with_empty(&a_empty, b_empty.len());
        let (ignore, early) = (ignore_per_row_errors, early_finite_check);
        Self::new_taking_as_empty(a, b, &a_as_empty, &b_as_empty, ignore, early)
    }

    /// See [`GeosPairs::new`], for an operation whose result is empty when `a` is, but not when `b` is.
    /// A row of `b` paired with an empty row of `a` is therefore not converted, and is `Empty` too.
    /// An empty row on either side (see [`rows_without_vertices`]) is `Empty`.
    pub(super) fn new_skipping_partners_of_empty_a(
        a: &GeometryArray,
        b: &GeometryArray,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<Self> {
        let (a_empty, b_empty) = (rows_without_vertices(a), rows_without_vertices(b));
        let b_as_empty = &b_empty | &paired_with_empty(&a_empty, b_empty.len());
        let (ignore, early) = (ignore_per_row_errors, early_finite_check);
        Self::new_taking_as_empty(a, b, &a_empty, &b_as_empty, ignore, early)
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

/// Marks the valid rows that have no vertices, which `is_empty` (without `check_nested`) calls empty.
/// A polygon with only rings without vertices is one of them.
/// A point has none when it is GeoArrow's empty point (NaN x and y).
fn rows_without_vertices(array: &GeometryArray) -> Bitmap {
    match array {
        GeometryArray::Point(array) => {
            let coords = &array.coords;
            let is_empty = |i| coords.xy(i).is_some_and(|(x, y)| x.is_nan() && y.is_nan());
            Bitmap::from_iter((0..coords.len()).map(is_empty))
        }
        GeometryArray::LineString(array) => {
            valid_rows_without_vertices(&array.rows, |vertices| vertices)
        }
        GeometryArray::MultiPoint(array) => {
            valid_rows_without_vertices(&array.rows, |points| points)
        }
        GeometryArray::Polygon(array) => {
            valid_rows_without_vertices(&array.rows, |rings| array.rings.span(rings))
        }
        GeometryArray::MultiLineString(array) => {
            valid_rows_without_vertices(&array.rows, |lines| array.lines.span(lines))
        }
        GeometryArray::MultiPolygon(array) => {
            valid_rows_without_vertices(&array.rows, |polygons| {
                array.rings.span(array.polygons.span(polygons))
            })
        }
    }
}

fn valid_rows_without_vertices(
    rows: &List,
    vertices: impl Fn(Range<usize>) -> Range<usize>,
) -> Bitmap {
    // A null row's offsets point anywhere, so its validity is checked first.
    Bitmap::from_iter(
        (0..rows.len()).map(|i| rows.is_valid(i) && vertices(rows.range(i)).is_empty()),
    )
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
