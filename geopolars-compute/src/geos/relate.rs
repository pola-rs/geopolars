//! The DE-9IM matrix of two geometries, forwarded to GEOS.
//! Implements the Python `relate`.

use geopolars_views::GeometryArray;
use geos::{Geom, Geometry};
use polars_arrow::array::{MutableBinaryViewArray, Utf8ViewArray};

use super::pairs::GeosPairs;
use super::GeosRow;
use crate::{row_or_null, EarlyCoordCheck, GeoComputeResult, OnRowError};

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

/// Two arrays of geometries converted to GEOS, with their rows paired up to relate.
/// A single row on either side is paired with every row of the other,
/// and stays on its own side, as swapping the two transposes their matrix.
pub struct RelatePairs {
    pairs: GeosPairs,
    /// Stands in for a row without parts, which never went to GEOS.
    empty: Geometry,
    on_row_error: OnRowError,
}

impl RelatePairs {
    /// Each side is converted to GEOS in full, so a row paired with a null can still fail.
    /// A row that fails here or in [`RelatePairs::matrices`] goes to `on_row_error`.
    ///
    /// `a` and `b` must have the same length, or one of them a single row.
    pub fn new(
        a: &GeometryArray,
        b: &GeometryArray,
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
    ) -> GeoComputeResult<Self> {
        // The matrix against an empty geometry still depends on its partner,
        // so unlike for `intersection`, that partner goes to GEOS too.
        Ok(Self {
            pairs: GeosPairs::new(a, b, on_row_error, early_coord_check)?,
            empty: Geometry::create_empty_point()?,
            on_row_error,
        })
    }

    /// Relates every pair lazily, as the iterator advances.
    pub fn matrices(&self) -> impl Iterator<Item = GeoComputeResult<Option<De9im>>> + '_ {
        self.pairs
            .iter()
            .map(|(a, b)| row_or_null(self.relate_rows(a, b), self.on_row_error))
    }

    #[inline(always)]
    fn relate_rows(
        &self,
        a: Option<&GeosRow>,
        b: Option<&GeosRow>,
    ) -> GeoComputeResult<Option<De9im>> {
        let (Some(a), Some(b)) = (self.relate_rows_geometry(a), self.relate_rows_geometry(b))
        else {
            return Ok(None);
        };
        De9im::between(a, b).map(Some)
    }

    #[inline(always)]
    fn relate_rows_geometry<'a>(&'a self, row: Option<&'a GeosRow>) -> Option<&'a Geometry> {
        match row? {
            GeosRow::Empty => Some(&self.empty),
            GeosRow::Geometry(geometry) | GeosRow::Unchanged(geometry) => Some(geometry),
        }
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
