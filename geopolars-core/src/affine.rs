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

/// Transforms every geometry in one chunk of a `kind` column's storage by `matrix`.
/// The result is the storage of the same kind and dimension.
pub fn affine_transform(
    chunk: &dyn Array,
    kind: Kind,
    matrix: &AffineMatrix,
) -> PolarsResult<Box<dyn Array>> {
    affine_transform_with_row_shift(chunk, kind, matrix, RowShift::None)
}

/// Transforms every geometry in one chunk of a `kind` column's storage by `matrix`,
/// about an origin of its own: see [`RowShift::About`].
///
/// `origins` is the storage of a point column, with one point per row of `chunk`.
pub fn affine_transform_about(
    chunk: &dyn Array,
    kind: Kind,
    matrix: &AffineMatrix,
    origins: &dyn Array,
) -> PolarsResult<Box<dyn Array>> {
    let origins = PointArray::new(Coords::new(origins)?);
    affine_transform_with_row_shift(chunk, kind, matrix, RowShift::About(&origins))
}

/// Transforms every geometry in one chunk of a `kind` column's storage by `matrix`,
/// then moves it by a vector of its own: see [`RowShift::Then`].
///
/// `vectors` is the storage of a point column, with one point per row of `chunk`.
pub fn affine_transform_then_shift(
    chunk: &dyn Array,
    kind: Kind,
    matrix: &AffineMatrix,
    vectors: &dyn Array,
) -> PolarsResult<Box<dyn Array>> {
    let vectors = PointArray::new(Coords::new(vectors)?);
    affine_transform_with_row_shift(chunk, kind, matrix, RowShift::Then(&vectors))
}

fn affine_transform_with_row_shift(
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
