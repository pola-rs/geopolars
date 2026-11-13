use geopolars_compute::native::{line_strings_are_rings, multi_line_strings_are_rings};
use geopolars_compute::with_const_flags;
use geopolars_views::{downcast, LineStringArray, MultiLineStringArray};
use polars_arrow::array::{Array, BooleanArray};
use polars_core::prelude::PolarsResult;

use crate::LineKind;

/// Checks whether every linestring in one chunk of a `kind` column's storage is a ring,
/// or whether the parts of every multilinestring chain into one.
/// Implements the Python `is_ring`.
pub fn are_rings(chunk: &dyn Array, kind: LineKind) -> PolarsResult<BooleanArray> {
    Ok(match kind {
        LineKind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            let z = array.coords().dim().has_z();
            with_const_flags!(Z = z; line_strings_are_rings::<Z>(&array))
        }
        LineKind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            let z = array.coords().dim().has_z();
            with_const_flags!(Z = z; multi_line_strings_are_rings::<Z>(&array))
        }
    })
}
