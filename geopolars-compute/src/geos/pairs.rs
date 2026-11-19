//! Pairing up the rows of two arrays of geometries, for a GEOS operation on two geometries.
//!
//! Each operation implements [`PairOperation`].
//! It says what GEOS makes of two geometries,
//! and what a pair gives when one or both of its rows are empty (see [`rows_without_vertices`]).
//! The result for one empty row is only asked for when the other row is not empty.
//!
//! [`GeosPairs`] uses this to convert only the rows the operation needs.
//! An empty row never goes to GEOS.
//! A row is empty when it has no vertices, as `is_empty` sees it.
//! Its partner only does when the operation needs it for the result ([`WhenEmpty::WithPartner`]).
//! A pair with a null row is null, so its other row is not converted either.
//!
//! A row that fails to convert (with `on_row_error="null"`) makes null only the pairs that need its geometry.
//! Its pair with an empty row still gives what [`WhenEmpty::Gives`] gives, whatever the chunking.

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
        Ok(Self {
            a: a.to_geos_rows_skipping(Some(&a_skipped), on_row_error, early)?,
            b: b.to_geos_rows_skipping(Some(&b_skipped), on_row_error, early)?,
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
        let a = broadcast(self.a.geometries.iter(), self.rows);
        let b = broadcast(self.b.geometries.iter(), self.rows);
        a.zip(b).map(move |(a, b)| {
            let result = match (a, b) {
                (GeosRow::Null, _) | (_, GeosRow::Null) => Ok(None),
                (GeosRow::Empty, GeosRow::Empty) => Ok(Some(operation.when_both_empty())),
                (GeosRow::Empty, b) => when_empty(operation, operation.when_a_empty(), b),
                (a, GeosRow::Empty) => when_empty(operation, operation.when_b_empty(), a),
                (GeosRow::Geometry { geometry: a, .. }, GeosRow::Geometry { geometry: b, .. }) => {
                    operation.on_geometries(a, b).map(Some)
                }
                (GeosRow::Failed, _) | (_, GeosRow::Failed) => Ok(None),
                (GeosRow::Skipped, _) | (_, GeosRow::Skipped) => {
                    unreachable!("a row with vertices that an operation needs is converted")
                }
            };
            row_or_null(result, on_row_error)
        })
    }

    pub(super) fn len(&self) -> usize {
        self.rows
    }

    pub(super) fn size_hints(&self) -> (SizeHint<2>, SizeHint<2>) {
        (self.a.size_hint, self.b.size_hint)
    }
}

/// `None` if the operation needs `partner`, which failed to convert.
#[inline(always)]
fn when_empty<Op: PairOperation>(
    operation: &Op,
    rule: WhenEmpty<Op>,
    partner: &GeosRow,
) -> GeoComputeResult<Option<Op::Output>> {
    match (rule, partner) {
        (WhenEmpty::Gives(result), _) => Ok(Some(result())),
        (WhenEmpty::WithPartner(result), GeosRow::Geometry { geometry, .. }) => {
            result(operation, geometry).map(Some)
        }
        (WhenEmpty::WithPartner(_), GeosRow::Failed) => Ok(None),
        (WhenEmpty::WithPartner(_), _) => {
            unreachable!("a row with vertices that an operation needs is converted")
        }
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
