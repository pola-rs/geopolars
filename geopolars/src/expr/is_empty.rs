use geopolars_core::are_empty;
use polars::prelude::*;
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

/// See `is_empty`.
#[polars_expr(output_type_func=kernel_output_type_is_empty)]
fn is_empty(inputs: &[Series], kwargs: IsEmptyKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| are_empty(chunk.as_ref(), kind, kwargs.check_nested));
    Ok(BooleanChunked::try_from_chunk_iter(inputs[0].name().clone(), chunks)?.into_series())
}
