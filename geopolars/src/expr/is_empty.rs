//! Whether a geometry is empty.
//! See `is_empty` in Python for the rules.

use geopolars_compute::native::{
    line_strings_are_empty, multi_line_strings_are_empty, multi_points_are_empty,
    multi_polygons_are_empty, points_are_empty, polygons_are_empty,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::describe;

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsEmptyKwargs {
    check_nested: bool,
}

/// For [`is_empty`]: a `bool` per geometry, of any kind.
fn kernel_output_type_is_empty(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// The kernel of `kind` on one chunk of the storage.
fn chunk_emptiness(
    chunk: &dyn Array,
    kind: Kind,
    check_nested: bool,
) -> PolarsResult<BooleanArray> {
    Ok(match kind {
        Kind::Point => points_are_empty(&PointArray::new(Coords::new(chunk)?)),
        Kind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(
                CHECK_NESTED = check_nested;
                line_strings_are_empty::<CHECK_NESTED>(&array)
            )
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(
                CHECK_NESTED = check_nested;
                multi_points_are_empty::<CHECK_NESTED>(&array)
            )
        }
        Kind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(
                CHECK_NESTED = check_nested;
                polygons_are_empty::<CHECK_NESTED>(&array)
            )
        }
        Kind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            with_const_flags!(
                CHECK_NESTED = check_nested;
                multi_line_strings_are_empty::<CHECK_NESTED>(&array)
            )
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(
                CHECK_NESTED = check_nested;
                multi_polygons_are_empty::<CHECK_NESTED>(&array)
            )
        }
    })
}

/// See `is_empty`.
#[polars_expr(output_type_func=kernel_output_type_is_empty)]
fn is_empty(inputs: &[Series], kwargs: IsEmptyKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_emptiness(chunk.as_ref(), kind, kwargs.check_nested));
    Ok(BooleanChunked::try_from_chunk_iter(inputs[0].name().clone(), chunks)?.into_series())
}
