use geopolars_compute::native::{
    line_strings_with_clamped_coords, multi_line_strings_with_clamped_coords,
    multi_points_with_clamped_coords, multi_polygons_with_clamped_coords,
    points_with_clamped_coords, polygons_with_clamped_coords, Bounds,
};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::{Array, ListArray};
use polars_core::prelude::PolarsResult;

/// Clamps the coordinates of every geometry in one chunk of a `kind` column's storage
/// to `bounds`.
/// Implements the Python `fix_out_of_bounds` with `action="clamp"`.
/// The result is the storage of the same kind and dimension.
pub fn with_clamped_coords(
    chunk: &dyn Array,
    kind: Kind,
    bounds: &Bounds,
) -> PolarsResult<Box<dyn Array>> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_with_clamped_coords(&PointArray::new(Coords::new(chunk)?), bounds)
            .into_struct_array()
            .boxed(),
        Kind::LineString => {
            line_strings_with_clamped_coords(&LineStringArray::new(rows()?)?, bounds)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPoint => {
            multi_points_with_clamped_coords(&MultiPointArray::new(rows()?)?, bounds)
                .into_list_array()
                .boxed()
        }
        Kind::Polygon => polygons_with_clamped_coords(&PolygonArray::new(rows()?)?, bounds)
            .into_list_array()
            .boxed(),
        Kind::MultiLineString => {
            multi_line_strings_with_clamped_coords(&MultiLineStringArray::new(rows()?)?, bounds)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            multi_polygons_with_clamped_coords(&MultiPolygonArray::new(rows()?)?, bounds)
                .into_list_array()
                .boxed()
        }
    })
}
