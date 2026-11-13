//! Sorting the parts of a geometry.
//! Implements the Python `sort_parts`.

use geopolars_compute::native::SortOptions;
use geopolars_core::NestedKind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, geometry_series};

/// `rings` of the Python `sort_parts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RingsKwarg {
    Keep,
    Rotate,
}

/// The keyword arguments of the Python `sort_parts`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortPartsKwargs {
    descending: bool,
    rings: RingsKwarg,
    ignore_per_row_errors: bool,
}

fn kernel_output_type_sort_parts(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    NestedKind::new("sort_parts", describe(field.dtype())?.kind)?;
    Ok(field.clone())
}

#[polars_expr(output_type_func=kernel_output_type_sort_parts)]
fn sort_parts(inputs: &[Series], kwargs: SortPartsKwargs) -> PolarsResult<Series> {
    let kind = NestedKind::new("sort_parts", describe(inputs[0].dtype())?.kind)?;
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
    let geo = describe(inputs[0].dtype())?.geo();
    geometry_series(inputs[0].name().clone(), geo, chunks)
}
