//! Whether a linestring is a ring (see [`geopolars_compute::native::line_strings_are_rings`]),
//! or whether the parts of a multilinestring chain into one
//! (see [`geopolars_compute::native::multi_line_strings_are_rings`]).

use geopolars_core::{are_rings, LineKind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::describe;

fn line_kind(field: &Field) -> PolarsResult<LineKind> {
    LineKind::new("is_ring", describe(field.dtype())?.kind)
}

/// For [`is_ring`]: a `bool` per row.
fn kernel_output_type_is_ring(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    line_kind(field)?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `is_ring`.
#[polars_expr(output_type_func=kernel_output_type_is_ring)]
fn is_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let kind = line_kind(&inputs[0].field())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| are_rings(chunk.as_ref(), kind));
    Ok(BooleanChunked::try_from_chunk_iter(inputs[0].name().clone(), chunks)?.into_series())
}
