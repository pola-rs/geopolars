//! Whether a geometry runs counter-clockwise, with a polygon's holes clockwise.
//! See `is_ccw` in Python for the rules.

use geopolars_compute::native::{
    line_strings_are_ccw, multi_line_strings_are_ccw, multi_polygons_are_ccw, polygons_are_ccw,
};
use geopolars_compute::with_const_flags;
use geopolars_core::{
    Kind, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::describe;

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsCcwKwargs {
    holes_cw: bool,
}

fn unoriented(kind: Kind) -> PolarsError {
    polars_err!(SchemaMismatch: "is_ccw does not accept a `{}` column", kind.name())
}

/// For [`is_ccw`]: a `bool` per geometry.
fn kernel_output_type_is_ccw(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match describe(field.dtype())?.kind {
        Kind::LineString | Kind::Polygon | Kind::MultiLineString | Kind::MultiPolygon => {
            Ok(Field::new(field.name().clone(), DataType::Boolean))
        }
        kind @ (Kind::Point | Kind::MultiPoint) => Err(unoriented(kind)),
    }
}

/// See `is_ccw`.
#[polars_expr(output_type_func=kernel_output_type_is_ccw)]
fn is_ccw(inputs: &[Series], kwargs: IsCcwKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();
    let chunks = storage.list()?.downcast_iter();
    let holes_cw = kwargs.holes_cw;
    let out = match kind {
        Kind::LineString => {
            let chunks = chunks.map(|rows| {
                let array = LineStringArray::new_untrimmed(rows)?;
                PolarsResult::Ok(line_strings_are_ccw(&array))
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::Polygon => {
            let chunks = chunks.map(|rows| {
                let array = PolygonArray::new_untrimmed(rows)?;
                PolarsResult::Ok(with_const_flags!(
                    HOLES_CW = holes_cw;
                    polygons_are_ccw::<HOLES_CW>(&array)
                ))
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiLineString => {
            let chunks = chunks.map(|rows| {
                let array = MultiLineStringArray::new_untrimmed(rows)?;
                PolarsResult::Ok(multi_line_strings_are_ccw(&array))
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiPolygon => {
            let chunks = chunks.map(|rows| {
                let array = MultiPolygonArray::new_untrimmed(rows)?;
                PolarsResult::Ok(with_const_flags!(
                    HOLES_CW = holes_cw;
                    multi_polygons_are_ccw::<HOLES_CW>(&array)
                ))
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        kind @ (Kind::Point | Kind::MultiPoint) => return Err(unoriented(kind)),
    };
    Ok(out)
}
