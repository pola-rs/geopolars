//! Sorting the parts of a geometry.
//! See `sort_parts` in Python for the rules.

use geopolars_compute::native::SortOptions;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::describe;

/// `rings` of `sort_parts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RingsKwarg {
    Keep,
    Rotate,
}

/// See `sort_parts`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortPartsKwargs {
    descending: bool,
    rings: RingsKwarg,
    ignore_per_row_errors: bool,
}

/// See `sort_parts`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn sort_parts(inputs: &[Series], kwargs: SortPartsKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let options = SortOptions {
        descending: kwargs.descending,
        rotate_rings: kwargs.rings == RingsKwarg::Rotate,
        ignore_per_row_errors: kwargs.ignore_per_row_errors,
    };
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| geopolars_core::sort_parts(chunk.as_ref(), kind, options))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
