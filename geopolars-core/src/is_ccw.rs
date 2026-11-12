use geopolars_compute::native::{
    line_strings_are_ccw, multi_line_strings_are_ccw, multi_polygons_are_ccw, polygons_are_ccw,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, Kind, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::{Array, BooleanArray};
use polars_core::prelude::PolarsResult;

use crate::refused;

/// Checks whether every (multi)linestring or (multi)polygon in one chunk of a `kind` column's storage
/// runs counter-clockwise.
/// Implements the Python `is_ccw`.
pub fn are_ccw(chunk: &dyn Array, kind: Kind, holes_cw: bool) -> PolarsResult<BooleanArray> {
    Ok(match kind {
        Kind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            line_strings_are_ccw(&array)
        }
        Kind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(HOLES_CW = holes_cw; polygons_are_ccw::<HOLES_CW>(&array))
        }
        Kind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            multi_line_strings_are_ccw(&array)
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(HOLES_CW = holes_cw; multi_polygons_are_ccw::<HOLES_CW>(&array))
        }
        Kind::Point | Kind::MultiPoint => return Err(refused("is_ccw", kind)),
    })
}
