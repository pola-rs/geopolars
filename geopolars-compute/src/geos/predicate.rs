//! Predicates on two geometries, such as whether they intersect, forwarded to GEOS.
//! See each predicate in Python for the rules.

use geopolars_core::GeometryArray;
use geos::{Geom, Geometry, PreparedGeometry};
use polars_arrow::array::BooleanArray;

use super::pairs::GeosPairs;
use super::GeosRow;
use crate::per_row::optional_bool_array;
use crate::{row_or_null, GeoComputeResult};

/// A predicate GEOS tests on two geometries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predicate {
    Intersects,
}

impl Predicate {
    /// Its result against an empty geometry, which never goes to GEOS.
    fn with_empty(self) -> bool {
        match self {
            Self::Intersects => false,
        }
    }

    /// The predicate that holds for `(b, a)` whenever this one holds for `(a, b)`.
    fn converse(self) -> Self {
        match self {
            Self::Intersects => Self::Intersects,
        }
    }

    #[inline(always)]
    fn test(self, a: &Geometry, b: &Geometry) -> GeoComputeResult<bool> {
        Ok(match self {
            Self::Intersects => a.intersects(b)?,
        })
    }

    #[inline(always)]
    fn test_prepared(self, a: &PreparedGeometry, b: &Geometry) -> GeoComputeResult<bool> {
        Ok(match self {
            Self::Intersects => a.intersects(b)?,
        })
    }
}

/// Tests `predicate` on two arrays of geometries, row by row.
/// Implements the Python predicates, such as `intersects`, which document their behaviour.
/// A single geometry on either side is tested against every row of the other,
/// and is prepared once for that.
/// Each side is converted to GEOS in full, except for the rows paired with an empty geometry.
/// A row paired with a null can therefore still fail.
///
/// `a` and `b` must have the same length, or one of them a single row.
pub fn test_predicate(
    a: &GeometryArray,
    b: &GeometryArray,
    predicate: Predicate,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<BooleanArray> {
    let pairs =
        GeosPairs::new_skipping_partners_of_empty(a, b, ignore_per_row_errors, early_finite_check)?;
    let tested = match pairs.sides() {
        ([GeosRow::Geometry(one)], many) if many.len() > 1 => {
            test_predicate_against_one(one, many, predicate, ignore_per_row_errors)
        }
        (many, [GeosRow::Geometry(one)]) if many.len() > 1 => {
            test_predicate_against_one(one, many, predicate.converse(), ignore_per_row_errors)
        }
        _ => pairs
            .iter()
            .map(|(a, b)| {
                row_or_null(
                    test_predicate_on_pair(a, b, predicate),
                    ignore_per_row_errors,
                )
            })
            .collect(),
    }?;
    Ok(optional_bool_array(&tested))
}

#[inline(always)]
fn test_predicate_on_pair(
    a: &GeosRow,
    b: &GeosRow,
    predicate: Predicate,
) -> GeoComputeResult<Option<bool>> {
    Ok(Some(match (a, b) {
        (GeosRow::Null, _) | (_, GeosRow::Null) => return Ok(None),
        (GeosRow::Empty, _) | (_, GeosRow::Empty) => predicate.with_empty(),
        (GeosRow::Geometry(a), GeosRow::Geometry(b)) => predicate.test(a, b)?,
    }))
}

/// Tests `predicate` with `one` on the left of every row of `many`.
fn test_predicate_against_one(
    one: &Geometry,
    many: &[GeosRow],
    predicate: Predicate,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<Vec<Option<bool>>> {
    let one = PreparedGeometry::new(one)?;
    let test = |row: &GeosRow| -> GeoComputeResult<Option<bool>> {
        Ok(Some(match row {
            GeosRow::Null => return Ok(None),
            GeosRow::Empty => predicate.with_empty(),
            GeosRow::Geometry(row) => predicate.test_prepared(&one, row)?,
        }))
    };
    many.iter()
        .map(|row| row_or_null(test(row), ignore_per_row_errors))
        .collect()
}
