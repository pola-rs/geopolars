use geopolars_compute::native::{
    line_strings_with_rings_rotated, multi_line_strings_sorted, multi_points_sorted,
    multi_polygons_sorted, polygons_with_holes_sorted, SortOptions,
};
use geopolars_views::{
    downcast, Kind, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::{polars_bail, PolarsResult};

use crate::{refused, IntoPolarsResult};

/// Sorts the parts of every geometry in one chunk of a `kind` column's storage.
/// Implements the Python `sort_parts`, which hands back a point, or a linestring it does not rotate,
/// without calling this.
/// The result is the storage of the same kind and dimension.
pub fn sort_parts(
    chunk: &dyn Array,
    kind: Kind,
    options: SortOptions,
) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        Kind::Point => return Err(refused("sort_parts", kind)),
        Kind::LineString if !options.rotate_rings => {
            polars_bail!(InvalidOperation: "sort_parts has no kernel for {kind:?} without rotating")
        }
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            line_strings_with_rings_rotated(&array, options.ignore_per_row_errors)
                .into_polars()?
                .into_list_array()
                .boxed()
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            multi_points_sorted(&array, options.descending)
                .into_list_array()
                .boxed()
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            polygons_with_holes_sorted(&array, options)
                .into_polars()?
                .into_list_array()
                .boxed()
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            multi_line_strings_sorted(&array, options)
                .into_polars()?
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            multi_polygons_sorted(&array, options)
                .into_polars()?
                .into_list_array()
                .boxed()
        }
    })
}
