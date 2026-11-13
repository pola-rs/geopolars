use geopolars_compute::native::{
    line_strings_to_ccw, multi_line_strings_to_ccw, multi_polygons_to_ccw, polygons_to_ccw,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, GeometryArray, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::SegmentKind;

/// Makes every (multi)linestring or (multi)polygon in one chunk of a `kind` column's storage
/// run counter-clockwise.
/// Implements the Python `to_ccw`.
/// The result is of the same kind and dimension.
pub fn to_ccw(chunk: &dyn Array, kind: SegmentKind, holes_cw: bool) -> PolarsResult<GeometryArray> {
    Ok(match kind {
        SegmentKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            GeometryArray::LineString(line_strings_to_ccw(&array))
        }
        SegmentKind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            GeometryArray::Polygon(
                with_const_flags!(HOLES_CW = holes_cw; polygons_to_ccw::<HOLES_CW>(&array)),
            )
        }
        SegmentKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            GeometryArray::MultiLineString(multi_line_strings_to_ccw(&array))
        }
        SegmentKind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            GeometryArray::MultiPolygon(
                with_const_flags!(HOLES_CW = holes_cw; multi_polygons_to_ccw::<HOLES_CW>(&array)),
            )
        }
    })
}
