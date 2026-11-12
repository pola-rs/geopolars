//! Whether a geometry is simple: natively for points and multipoints,
//! through `geopolars_compute::geos` for the others.
//! See `is_simple` in Python for the rules.

use geopolars_core::are_simple;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::describe;

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
        .map(|chunk| are_simple(chunk.as_ref(), kind));
    Ok(BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)?.into_series())
}
