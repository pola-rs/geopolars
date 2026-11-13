use geopolars_compute::native::{
    line_string_coordinate_counts, multi_line_string_coordinate_counts,
    multi_point_coordinate_counts, multi_polygon_coordinate_counts, point_coordinate_counts,
    polygon_coordinate_counts,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::{Array, PrimitiveArray};
use polars_core::prelude::PolarsResult;

/// Counts the coordinates of every geometry in one chunk of a `kind` column's storage.
/// Implements the Python `count_coordinates`.
pub fn coordinate_counts(
    chunk: &dyn Array,
    kind: Kind,
    include_empty: bool,
) -> PolarsResult<PrimitiveArray<u32>> {
    Ok(match kind {
        Kind::Point => {
            let array = PointArray::new(Coords::new(chunk)?);
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                point_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
        Kind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                line_string_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                multi_point_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
        Kind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                polygon_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
        Kind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                multi_line_string_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                multi_polygon_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
    })
}
