use geopolars_compute::native::{
    line_strings_with_rings_rotated, multi_line_strings_sorted, multi_points_sorted,
    multi_polygons_sorted, polygons_with_holes_sorted, SortOptions,
};
use geopolars_views::{
    downcast, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::{polars_bail, PolarsResult};

use crate::{IntoPolarsResult, NestedKind};

/// Sorts the parts of every geometry in one chunk of a `kind` column's storage.
/// Implements the Python `sort_parts`.
/// The result is the storage of the same kind and dimension.
///
/// A linestring is refused unless `options.rotate_rings=true`.
/// The Python `sort_parts` hands such a linestring back without calling this, like a point.
pub fn sort_parts(
    chunk: &dyn Array,
    kind: NestedKind,
    options: SortOptions,
) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        NestedKind::LineString if !options.rotate_rings => {
            polars_bail!(InvalidOperation: "sort_parts has no kernel for {kind:?} without rotating")
        }
        NestedKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            line_strings_with_rings_rotated(&array, options.ignore_per_row_errors)
                .into_polars()?
                .into_list_array()
                .boxed()
        }
        NestedKind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            multi_points_sorted(&array, options.descending)
                .into_list_array()
                .boxed()
        }
        NestedKind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            polygons_with_holes_sorted(&array, options)
                .into_polars()?
                .into_list_array()
                .boxed()
        }
        NestedKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            multi_line_strings_sorted(&array, options)
                .into_polars()?
                .into_list_array()
                .boxed()
        }
        NestedKind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            multi_polygons_sorted(&array, options)
                .into_polars()?
                .into_list_array()
                .boxed()
        }
    })
}
