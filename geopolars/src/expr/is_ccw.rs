//! Whether a geometry runs counter-clockwise.
//! Implements the Python `is_ccw`.

use geopolars_core::{are_ccw, SegmentKind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::describe;

/// The keyword arguments of the Python `is_ccw`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsCcwKwargs {
    holes_cw: bool,
}

fn kernel_output_type_is_ccw(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    SegmentKind::new("is_ccw", describe(field.dtype())?.kind)?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

#[polars_expr(output_type_func=kernel_output_type_is_ccw)]
fn is_ccw(inputs: &[Series], kwargs: IsCcwKwargs) -> PolarsResult<Series> {
    let kind = SegmentKind::new("is_ccw", describe(inputs[0].dtype())?.kind)?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| are_ccw(chunk.as_ref(), kind, kwargs.holes_cw));
    Ok(BooleanChunked::try_from_chunk_iter(inputs[0].name().clone(), chunks)?.into_series())
}
