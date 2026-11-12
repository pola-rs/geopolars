use geopolars_compute::native::{
    affine_line_strings, affine_multi_line_strings, affine_multi_points, affine_multi_polygons,
    affine_points, affine_polygons, AffineMatrix, RowShift,
};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

/// Transforms every geometry in one chunk of a `kind` column's storage by `matrix`,
/// and moves it by `shift`.
/// The result is the storage of the same kind and dimension.
pub fn affine_transform(
    chunk: &dyn Array,
    kind: Kind,
    matrix: &AffineMatrix,
    shift: RowShift,
) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        Kind::Point => {
            let array = PointArray::new(Coords::new(chunk)?);
            affine_points(matrix, &array, shift)
                .into_struct_array()
                .boxed()
        }
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            affine_line_strings(matrix, &array, shift)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            affine_multi_points(matrix, &array, shift)
                .into_list_array()
                .boxed()
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            affine_polygons(matrix, &array, shift)
                .into_list_array()
                .boxed()
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            affine_multi_line_strings(matrix, &array, shift)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            affine_multi_polygons(matrix, &array, shift)
                .into_list_array()
                .boxed()
        }
    })
}
