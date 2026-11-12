use geopolars_compute::native::{
    reverse_line_strings, reverse_multi_line_strings, reverse_multi_points, reverse_multi_polygons,
    reverse_polygons,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, Kind, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::refused;

/// Reverses the order of the vertices of every geometry in one chunk of a `kind` column's storage.
/// Implements the Python `reverse`.
/// The result is the storage of the same kind and dimension.
pub fn reverse(
    chunk: &dyn Array,
    kind: Kind,
    reverse_holes: bool,
    nested: bool,
) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        Kind::Point => return Err(refused("reverse", kind)),
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            reverse_line_strings(&array).into_list_array().boxed()
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            reverse_multi_points(&array).into_list_array().boxed()
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(
                REVERSE_HOLES = reverse_holes;
                reverse_polygons::<REVERSE_HOLES>(&array)
            )
            .into_list_array()
            .boxed()
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            with_const_flags!(NESTED = nested; reverse_multi_line_strings::<NESTED>(&array))
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(
                REVERSE_HOLES = reverse_holes, NESTED = nested;
                reverse_multi_polygons::<REVERSE_HOLES, NESTED>(&array)
            )
            .into_list_array()
            .boxed()
        }
    })
}
