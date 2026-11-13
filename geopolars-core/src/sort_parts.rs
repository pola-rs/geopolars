use geopolars_compute::native::{
    line_strings_with_rings_rotated, multi_line_strings_sorted, multi_points_sorted,
    multi_polygons_sorted, polygons_with_holes_sorted, SortOptions,
};
use geopolars_views::{
    downcast, GeometryArray, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::{polars_bail, PolarsResult};

use crate::{IntoPolarsResult, NestedKind};

/// Sorts the parts of every geometry in one chunk of a `kind` column's storage.
/// Implements the Python `sort_parts`.
/// The result is of the same kind and dimension.
///
/// A linestring is refused unless `options.rotate_rings=true`.
/// The Python `sort_parts` hands such a linestring back without calling this, like a point.
pub fn sort_parts(
    chunk: &dyn Array,
    kind: NestedKind,
    options: SortOptions,
) -> PolarsResult<GeometryArray> {
    Ok(match kind {
        NestedKind::LineString if !options.rotate_rings => {
            polars_bail!(InvalidOperation: "sort_parts has no kernel for {kind:?} without rotating")
        }
        NestedKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            GeometryArray::LineString(
                line_strings_with_rings_rotated(&array, options.ignore_per_row_errors)
                    .into_polars()?,
            )
        }
        NestedKind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            GeometryArray::MultiPoint(multi_points_sorted(&array, options.descending))
        }
        NestedKind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            GeometryArray::Polygon(polygons_with_holes_sorted(&array, options).into_polars()?)
        }
        NestedKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            GeometryArray::MultiLineString(
                multi_line_strings_sorted(&array, options).into_polars()?,
            )
        }
        NestedKind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            GeometryArray::MultiPolygon(multi_polygons_sorted(&array, options).into_polars()?)
        }
    })
}
