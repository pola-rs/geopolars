use geopolars_compute::native::{line_strings_are_rings, multi_line_strings_are_rings};
use geopolars_compute::with_const_flags;
use geopolars_views::{downcast, Kind, LineStringArray, MultiLineStringArray};
use polars_arrow::array::{Array, BooleanArray};
use polars_core::prelude::PolarsResult;

use crate::refused;

/// Checks whether every linestring in one chunk of a `kind` column's storage is a ring,
/// or whether the parts of every multilinestring chain into one.
/// Implements the Python `is_ring`.
pub fn are_rings(chunk: &dyn Array, kind: Kind) -> PolarsResult<BooleanArray> {
    Ok(match kind {
        Kind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            let z = array.coords().dim().has_z();
            with_const_flags!(Z = z; line_strings_are_rings::<Z>(&array))
        }
        Kind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            let z = array.coords().dim().has_z();
            with_const_flags!(Z = z; multi_line_strings_are_rings::<Z>(&array))
        }
        Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon => {
            return Err(refused("is_ring", kind))
        }
    })
}
