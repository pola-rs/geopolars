//! Reversing the order of a geometry's vertices.
//! See `reverse` in Python for the rules.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::describe;

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReverseKwargs {
    reverse_holes: bool,
    nested: bool,
}

/// See `reverse`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn reverse(inputs: &[Series], kwargs: ReverseKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            geopolars_core::reverse(chunk.as_ref(), kind, kwargs.reverse_holes, kwargs.nested)
        })
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
