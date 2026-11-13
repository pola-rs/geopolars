use geopolars_compute::native::{
    reverse_line_strings, reverse_multi_line_strings, reverse_multi_points, reverse_multi_polygons,
    reverse_polygons,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, GeometryArray, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::NestedKind;

/// Reverses the order of the vertices of every geometry,
/// for one chunk of a `kind` column's storage.
/// Implements the Python `reverse`.
/// The result is of the same kind and dimension.
pub fn reverse(
    chunk: &dyn Array,
    kind: NestedKind,
    reverse_holes: bool,
    nested: bool,
) -> PolarsResult<GeometryArray> {
    Ok(match kind {
        NestedKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            GeometryArray::LineString(reverse_line_strings(&array))
        }
        NestedKind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            GeometryArray::MultiPoint(reverse_multi_points(&array))
        }
        NestedKind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(
                REVERSE_HOLES = reverse_holes;
                GeometryArray::Polygon(reverse_polygons::<REVERSE_HOLES>(&array)
            ))
        }
        NestedKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            GeometryArray::MultiLineString(
                with_const_flags!(NESTED = nested; reverse_multi_line_strings::<NESTED>(&array)),
            )
        }
        NestedKind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(
                REVERSE_HOLES = reverse_holes, NESTED = nested;
                GeometryArray::MultiPolygon(reverse_multi_polygons::<REVERSE_HOLES, NESTED>(&array)
            ))
        }
    })
}
