//! Whether a geometry is simple: natively for points and multipoints,
//! through `geopolars_compute::geos` for the others.
//! See `is_simple` in Python for the rules.

use geopolars_compute::geos::{
    line_strings_are_simple, multi_line_strings_are_simple, multi_polygons_are_simple,
    polygons_are_simple,
};
use geopolars_compute::native::{multi_points_are_simple, points_are_simple};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use pyo3_polars::derive::polars_expr;

use super::IntoPolarsResult;
use crate::geoarrow::describe;

/// [`is_simple`] for one chunk of the storage of a `kind`.
fn chunk_is_simple(chunk: &dyn Array, kind: Kind) -> PolarsResult<BooleanArray> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_are_simple(&PointArray::new(Coords::new(chunk)?)),
        Kind::MultiPoint => multi_points_are_simple(&MultiPointArray::new(rows()?)?),
        Kind::LineString => {
            line_strings_are_simple(&LineStringArray::new(rows()?)?).into_polars()?
        }
        Kind::MultiLineString => {
            multi_line_strings_are_simple(&MultiLineStringArray::new(rows()?)?).into_polars()?
        }
        Kind::Polygon => polygons_are_simple(&PolygonArray::new(rows()?)?).into_polars()?,
        Kind::MultiPolygon => {
            multi_polygons_are_simple(&MultiPolygonArray::new(rows()?)?).into_polars()?
        }
    })
}

/// For [`is_simple`]: a `bool` per geometry.
fn kernel_output_type_is_simple(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `is_simple`.
#[polars_expr(output_type_func=kernel_output_type_is_simple)]
fn is_simple(inputs: &[Series]) -> PolarsResult<Series> {
    let geometries = &inputs[0];
    let kind = describe(geometries.dtype())?.kind;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_is_simple(chunk.as_ref(), kind));
    Ok(BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)?.into_series())
}
