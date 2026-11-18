//! The DE-9IM matrix of two geometries, forwarded to GEOS.
//! Implements the Python `relate`.

use geopolars_views::GeometryArray;
use geos::{Geom, Geometry};
use polars_arrow::array::{MutableBinaryViewArray, Utf8ViewArray};

use super::pairs::{GeosPairs, PairOperation, WhenEmpty};
use crate::{EarlyCoordCheck, GeoComputeResult, OnRowError};

/// A DE-9IM matrix, as GEOS writes it.
/// The Python `relate` documents its rows, columns, and cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct De9im([u8; 9]);

impl De9im {
    pub(crate) fn between(a: &impl Geom, b: &impl Geom) -> GeoComputeResult<Self> {
        let matrix = a.relate(b)?;
        let cells = matrix.as_bytes().try_into().map_err(|_| {
            geos::Error::GenericError(format!("expected a DE-9IM matrix, got {matrix:?}"))
        })?;
        Ok(Self(cells))
    }

    /// The matrix as text, row after row, such as `"212101212"`.
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).expect("GEOS writes a matrix in ASCII")
    }
}

/// See [`RelatePairs`].
/// The matrix against an empty row still depends on its partner,
/// so unlike for `intersection`, that partner goes to GEOS too.
struct RelateOperation {
    /// Stands in for an empty row, which never went to GEOS.
    empty: Geometry,
}

impl PairOperation for RelateOperation {
    type Output = De9im;

    fn when_both_empty(&self) -> De9im {
        // Only their exteriors meet, in an area.
        De9im(*b"FFFFFFFF2")
    }

    fn when_a_empty(&self) -> WhenEmpty<Self> {
        WhenEmpty::WithPartner(|relate, b| De9im::between(&relate.empty, b))
    }

    fn when_b_empty(&self) -> WhenEmpty<Self> {
        WhenEmpty::WithPartner(|relate, a| De9im::between(a, &relate.empty))
    }

    fn on_geometries(&self, a: &Geometry, b: &Geometry) -> GeoComputeResult<De9im> {
        De9im::between(a, b)
    }
}

/// Two arrays of geometries converted to GEOS, with their rows paired up to relate.
/// A single row on either side is paired with every row of the other,
/// and stays on its own side, as swapping the two transposes their matrix.
pub struct RelatePairs {
    pairs: GeosPairs,
    operation: RelateOperation,
    on_row_error: OnRowError,
}

impl RelatePairs {
    /// A row that fails here or in [`RelatePairs::matrices`] goes to `on_row_error`.
    ///
    /// `a` and `b` must have the same length, or one of them a single row.
    pub fn new(
        a: &GeometryArray,
        b: &GeometryArray,
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
    ) -> GeoComputeResult<Self> {
        let operation = RelateOperation {
            empty: Geometry::create_empty_point()?,
        };
        Ok(Self {
            pairs: GeosPairs::new(a, b, &operation, on_row_error, early_coord_check)?,
            operation,
            on_row_error,
        })
    }

    /// Relates every pair lazily, as the iterator advances.
    pub fn matrices(&self) -> impl Iterator<Item = GeoComputeResult<Option<De9im>>> + '_ {
        self.pairs.evaluate(&self.operation, self.on_row_error)
    }

    /// Implements the Python `relate`.
    pub fn to_strings(&self) -> GeoComputeResult<Utf8ViewArray> {
        let mut strings = MutableBinaryViewArray::<str>::with_capacity(self.pairs.len());
        for matrix in self.matrices() {
            strings.push(matrix?.as_ref().map(De9im::as_str));
        }
        Ok(strings.freeze())
    }
}
