//! Reversing the order of a geometry's vertices.
//! See `reverse` in Python for the rules.

use geopolars_core::NestedKind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, geometry_series};

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReverseKwargs {
    reverse_holes: bool,
    nested: bool,
}

/// For [`reverse`].
/// This gives the input's type.
fn kernel_output_type_reverse(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    NestedKind::new("reverse", describe(field.dtype())?.kind)?;
    Ok(field.clone())
}

/// See `reverse`.
#[polars_expr(output_type_func=kernel_output_type_reverse)]
fn reverse(inputs: &[Series], kwargs: ReverseKwargs) -> PolarsResult<Series> {
    let kind = NestedKind::new("reverse", describe(inputs[0].dtype())?.kind)?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            geopolars_core::reverse(chunk.as_ref(), kind, kwargs.reverse_holes, kwargs.nested)
        })
        .collect::<PolarsResult<_>>()?;
    let geo = describe(inputs[0].dtype())?.geo();
    geometry_series(inputs[0].name().clone(), geo, chunks)
}
