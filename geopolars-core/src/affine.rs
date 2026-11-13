use geopolars_compute::native::{
    affine_line_strings, affine_multi_line_strings, affine_multi_points, affine_multi_polygons,
    affine_points, affine_polygons, AffineMatrix, RowShift,
};
use geopolars_views::{
    downcast, Coords, GeometryArray, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

/// Transforms every geometry in one chunk of a `kind` column's storage by `matrix`.
/// Implements the Python `affine_transform` with an origin given as numbers.
/// The result is of the same kind and dimension.
pub fn affine_transform(
    chunk: &dyn Array,
    kind: Kind,
    matrix: &AffineMatrix,
) -> PolarsResult<GeometryArray> {
    affine_transform_with_row_shift(chunk, kind, matrix, RowShift::None)
}

/// Transforms every geometry like [`affine_transform`], but about an origin of its own
/// (see [`RowShift::About`]).
/// Implements the Python `affine_transform` with an origin given as a point column.
///
/// `origins` is the storage of a point column, with one point per row of `chunk`.
pub fn affine_transform_about(
    chunk: &dyn Array,
    kind: Kind,
    matrix: &AffineMatrix,
    origins: &dyn Array,
) -> PolarsResult<GeometryArray> {
    let origins = PointArray::new(Coords::new(origins)?);
    affine_transform_with_row_shift(chunk, kind, matrix, RowShift::About(&origins))
}

/// Transforms every geometry like [`affine_transform`], then moves it by a vector of its own
/// (see [`RowShift::Then`]).
/// Implements the Python `translate` with an offset given as columns.
///
/// `vectors` is the storage of a point column, with one point per row of `chunk`.
pub fn affine_transform_then_shift(
    chunk: &dyn Array,
    kind: Kind,
    matrix: &AffineMatrix,
    vectors: &dyn Array,
) -> PolarsResult<GeometryArray> {
    let vectors = PointArray::new(Coords::new(vectors)?);
    affine_transform_with_row_shift(chunk, kind, matrix, RowShift::Then(&vectors))
}

fn affine_transform_with_row_shift(
    chunk: &dyn Array,
    kind: Kind,
    matrix: &AffineMatrix,
    shift: RowShift,
) -> PolarsResult<GeometryArray> {
    Ok(match kind {
        Kind::Point => {
            let array = PointArray::new(Coords::new(chunk)?);
            GeometryArray::Point(affine_points(matrix, &array, shift))
        }
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            GeometryArray::LineString(affine_line_strings(matrix, &array, shift))
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            GeometryArray::MultiPoint(affine_multi_points(matrix, &array, shift))
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            GeometryArray::Polygon(affine_polygons(matrix, &array, shift))
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            GeometryArray::MultiLineString(affine_multi_line_strings(matrix, &array, shift))
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            GeometryArray::MultiPolygon(affine_multi_polygons(matrix, &array, shift))
        }
    })
}
