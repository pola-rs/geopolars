use geopolars_compute::native::{
    line_strings_are_ccw, multi_line_strings_are_ccw, multi_polygons_are_ccw, polygons_are_ccw,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::{Array, BooleanArray};
use polars_core::prelude::PolarsResult;

use crate::SegmentKind;

/// Checks whether every (multi)linestring or (multi)polygon in one chunk of a `kind` column's storage
/// runs counter-clockwise.
/// Implements the Python `is_ccw`.
pub fn are_ccw(chunk: &dyn Array, kind: SegmentKind, holes_cw: bool) -> PolarsResult<BooleanArray> {
    Ok(match kind {
        SegmentKind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            line_strings_are_ccw(&array)
        }
        SegmentKind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(HOLES_CW = holes_cw; polygons_are_ccw::<HOLES_CW>(&array))
        }
        SegmentKind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            multi_line_strings_are_ccw(&array)
        }
        SegmentKind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(HOLES_CW = holes_cw; multi_polygons_are_ccw::<HOLES_CW>(&array))
        }
    })
}
