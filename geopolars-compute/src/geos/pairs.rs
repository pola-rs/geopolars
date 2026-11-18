//! Pairing up the rows of two arrays of geometries, for a GEOS operation on two geometries.
//!
//! Each operation implements [`PairOperation`].
//! It says what GEOS makes of two geometries,
//! and what a pair gives when one or both of its rows are empty (see [`rows_without_vertices`]).
//! The result for one empty row is only asked for when the other row is not empty.
//!
//! [`GeosPairs`] uses this to convert only the rows the operation needs.
//! An empty row never goes to GEOS.
//! Its partner only does when the operation needs it for the result ([`WhenEmpty::WithPartner`]).
//! A pair with a null row is null, so its other row is not converted either.
//! An operation is therefore never handed a row that was not converted.

use geopolars_views::output::SizeHint;
use geopolars_views::GeometryArray;
use geos::Geometry;
use polars_arrow::bitmap::Bitmap;

use super::{rows_without_vertices, GeosGeometries, GeosRow, ToGeosRows};
use crate::{row_or_null, EarlyCoordCheck, GeoComputeResult, OnRowError};

/// A GEOS operation on two geometries, row by row.
pub(super) trait PairOperation {
    type Output;

    fn when_both_empty(&self) -> Self::Output;

    fn when_a_empty(&self) -> WhenEmpty<Self>;

    fn when_b_empty(&self) -> WhenEmpty<Self>;

    fn on_geometries(&self, a: &Geometry, b: &Geometry) -> GeoComputeResult<Self::Output>;
}

/// What a pair gives when one of its rows is empty.
pub(super) enum WhenEmpty<Op: PairOperation + ?Sized> {
    /// A result without the partner, which is therefore never converted.
    Gives(fn() -> Op::Output),
    /// A result from the partner, which is converted.
    WithPartner(fn(&Op, &Geometry) -> GeoComputeResult<Op::Output>),
}

/// Two arrays of geometries converted to GEOS, with their rows paired up.
/// A single row on either side is paired with every row of the other,
/// and stays on its own side, for an operation that is not symmetric.
pub(super) struct GeosPairs {
    // `b` is declared first so it drops first, freeing the geometries in reverse of building them.
    // With glibc's malloc, that makes `is_topologically_equal` on 50k polygons ~20% faster.
    b: GeosGeometries,
    a: GeosGeometries,
    /// The rows of `a` without vertices.
    a_empty: Bitmap,
    /// The rows of `b` without vertices.
    b_empty: Bitmap,
    rows: usize,
}

impl GeosPairs {
    /// Converts the rows of `a` and `b` that `operation` needs (see the module doc).
    /// A single row is only left out when every row it is paired with leaves it out.
    ///
    /// `a` and `b` must have the same length, or one of them a single row.
    pub(super) fn new<Op: PairOperation>(
        a: &GeometryArray,
        b: &GeometryArray,
        operation: &Op,
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
    ) -> GeoComputeResult<Self> {
        let rows = pair_count(a.len(), b.len());
        let (a_empty, b_empty) = (rows_without_vertices(a), rows_without_vertices(b));
        let decides = |array: &GeometryArray, empty: &Bitmap, rule: WhenEmpty<Op>| match rule {
            WhenEmpty::Gives(_) => &null_rows(array) | empty,
            WhenEmpty::WithPartner(_) => null_rows(array),
        };
        let a_decides = decides(a, &a_empty, operation.when_a_empty());
        let b_decides = decides(b, &b_empty, operation.when_b_empty());
        let a_skipped = &a_empty | &paired_with(&b_decides, a.len());
        let b_skipped = &b_empty | &paired_with(&a_decides, b.len());
        let early = early_coord_check;
        let a_rows = a.to_geos_rows_taking_as_empty(Some(&a_skipped), on_row_error, early)?;
        let b_rows = b.to_geos_rows_taking_as_empty(Some(&b_skipped), on_row_error, early)?;
        Ok(Self {
            b: b_rows,
            a: a_rows,
            a_empty,
            b_empty,
            rows,
        })
    }

    /// The result of every pair (see the module doc), lazily, as the iterator advances.
    /// A pair that fails goes to `on_row_error`.
    pub(super) fn evaluate<'a, Op: PairOperation>(
        &'a self,
        operation: &'a Op,
        on_row_error: OnRowError,
    ) -> impl Iterator<Item = GeoComputeResult<Option<Op::Output>>> + 'a {
        let a = broadcast(self.a.geometries.iter(), self.rows)
            .zip(broadcast(self.a_empty.iter(), self.rows));
        let b = broadcast(self.b.geometries.iter(), self.rows)
            .zip(broadcast(self.b_empty.iter(), self.rows));
        a.zip(b).map(move |((a, a_empty), (b, b_empty))| {
            let (Some(a), Some(b)) = (a, b) else {
                return Ok(None);
            };
            let result = match (a_empty, b_empty) {
                (true, true) => Ok(operation.when_both_empty()),
                (true, false) => when_empty(operation, operation.when_a_empty(), b),
                (false, true) => when_empty(operation, operation.when_b_empty(), a),
                (false, false) => operation.on_geometries(converted(a), converted(b)),
            };
            row_or_null(result.map(Some), on_row_error)
        })
    }

    pub(super) fn len(&self) -> usize {
        self.rows
    }

    pub(super) fn size_hints(&self) -> (SizeHint<2>, SizeHint<2>) {
        (self.a.size_hint, self.b.size_hint)
    }
}

#[inline(always)]
fn when_empty<Op: PairOperation>(
    operation: &Op,
    rule: WhenEmpty<Op>,
    partner: &GeosRow,
) -> GeoComputeResult<Op::Output> {
    match rule {
        WhenEmpty::Gives(result) => Ok(result()),
        WhenEmpty::WithPartner(result) => result(operation, converted(partner)),
    }
}

/// The GEOS geometry of a row that [`GeosPairs::new`] converted.
#[inline(always)]
fn converted(row: &GeosRow) -> &Geometry {
    match row {
        GeosRow::Geometry { geometry, .. } => geometry,
        GeosRow::Empty => unreachable!("a row with vertices that an operation needs is converted"),
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

/// Marks which of `len` rows are paired with a marked row of `partner_marked`,
/// broadcast as in [`GeosPairs::evaluate`].
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

/// Gives `len` items, repeating `items` if it holds a single one.
///
/// `items` must hold `len` items, or a single one.
fn broadcast<T>(items: impl Iterator<Item = T> + Clone, len: usize) -> impl Iterator<Item = T> {
    items.cycle().take(len)
}
