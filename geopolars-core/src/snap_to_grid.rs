use geopolars_compute::native::{
    line_strings_snapped_to_grid, multi_line_strings_snapped_to_grid, multi_points_snapped_to_grid,
    multi_polygons_snapped_to_grid, points_snapped_to_grid, polygons_snapped_to_grid, Grid,
};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::{Array, ListArray};
use polars_core::prelude::PolarsResult;

/// Snaps every coordinate of every geometry onto `grid`,
/// for one chunk of a `kind` column's storage.
/// Implements the Python `snap_to_grid`.
/// The result is the storage of the same kind and dimension.
pub fn snap_to_grid(chunk: &dyn Array, kind: Kind, grid: &Grid) -> PolarsResult<Box<dyn Array>> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_snapped_to_grid(&PointArray::new(Coords::new(chunk)?), grid)
            .into_struct_array()
            .boxed(),
        Kind::LineString => line_strings_snapped_to_grid(&LineStringArray::new(rows()?)?, grid)
            .into_list_array()
            .boxed(),
        Kind::MultiPoint => multi_points_snapped_to_grid(&MultiPointArray::new(rows()?)?, grid)
            .into_list_array()
            .boxed(),
        Kind::Polygon => polygons_snapped_to_grid(&PolygonArray::new(rows()?)?, grid)
            .into_list_array()
            .boxed(),
        Kind::MultiLineString => {
            multi_line_strings_snapped_to_grid(&MultiLineStringArray::new(rows()?)?, grid)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            multi_polygons_snapped_to_grid(&MultiPolygonArray::new(rows()?)?, grid)
                .into_list_array()
                .boxed()
        }
    })
}
