use geopolars_compute::native::{
    line_strings_to_ccw, multi_line_strings_to_ccw, multi_polygons_to_ccw, polygons_to_ccw,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::SegmentKind;

/// Makes every (multi)linestring or (multi)polygon in one chunk of a `kind` column's storage
/// run counter-clockwise.
/// Implements the Python `to_ccw`.
/// The result is the storage of the same kind and dimension.
pub fn to_ccw(
    chunk: &dyn Array,
    kind: SegmentKind,
    holes_cw: bool,
) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        SegmentKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            line_strings_to_ccw(&array).into_list_array().boxed()
        }
        SegmentKind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(HOLES_CW = holes_cw; polygons_to_ccw::<HOLES_CW>(&array))
                .into_list_array()
                .boxed()
        }
        SegmentKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            multi_line_strings_to_ccw(&array).into_list_array().boxed()
        }
        SegmentKind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(HOLES_CW = holes_cw; multi_polygons_to_ccw::<HOLES_CW>(&array))
                .into_list_array()
                .boxed()
        }
    })
}
