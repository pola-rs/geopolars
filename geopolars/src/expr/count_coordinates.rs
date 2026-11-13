use geopolars_core::coordinate_counts;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::describe;

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountCoordinatesKwargs {
    include_empty: bool,
}

/// For [`count_coordinates`]: a `u32` per geometry, of any kind.
fn kernel_output_type_count_coordinates(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::UInt32))
}

/// See `count_coordinates`.
#[polars_expr(output_type_func=kernel_output_type_count_coordinates)]
fn count_coordinates(inputs: &[Series], kwargs: CountCoordinatesKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| coordinate_counts(chunk.as_ref(), kind, kwargs.include_empty));
    Ok(UInt32Chunked::try_from_chunk_iter(inputs[0].name().clone(), chunks)?.into_series())
}
