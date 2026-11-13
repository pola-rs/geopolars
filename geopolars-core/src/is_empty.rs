use geopolars_compute::native::{
    line_strings_are_empty, multi_line_strings_are_empty, multi_points_are_empty,
    multi_polygons_are_empty, points_are_empty, polygons_are_empty,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::{Array, BooleanArray};
use polars_core::prelude::PolarsResult;

/// Checks whether every geometry in one chunk of a `kind` column's storage is empty.
/// Implements the Python `is_empty`.
pub fn are_empty(chunk: &dyn Array, kind: Kind, check_nested: bool) -> PolarsResult<BooleanArray> {
    Ok(match kind {
        Kind::Point => points_are_empty(&PointArray::new(Coords::new(chunk)?)),
        Kind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(
                CHECK_NESTED = check_nested;
                line_strings_are_empty::<CHECK_NESTED>(&array)
            )
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(
                CHECK_NESTED = check_nested;
                multi_points_are_empty::<CHECK_NESTED>(&array)
            )
        }
        Kind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(
                CHECK_NESTED = check_nested;
                polygons_are_empty::<CHECK_NESTED>(&array)
            )
        }
        Kind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            with_const_flags!(
                CHECK_NESTED = check_nested;
                multi_line_strings_are_empty::<CHECK_NESTED>(&array)
            )
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(
                CHECK_NESTED = check_nested;
                multi_polygons_are_empty::<CHECK_NESTED>(&array)
            )
        }
    })
}
