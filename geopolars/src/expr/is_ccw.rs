//! Whether a geometry runs counter-clockwise, with a polygon's holes clockwise.
//! See `is_ccw` in Python for the rules.

use geopolars_core::are_ccw;
use geopolars_views::Kind;
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
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| are_ccw(chunk.as_ref(), kind, kwargs.holes_cw));
    Ok(BooleanChunked::try_from_chunk_iter(inputs[0].name().clone(), chunks)?.into_series())
}
