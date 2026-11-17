//! The DE-9IM matrix of two geometries
//! Implements the Python `relate`.

use crate::common_arguments::{EarlyCoordCheckDef, OnRowErrorDef};
use geopolars_compute::{EarlyCoordCheck, OnRowError};
use polars::prelude::*;
use polars_arrow::array::Utf8ViewArray;
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{
    describe, ensure_same_crs, kind_of, single_chunk_masked_pairs, GeometryPairs,
};

/// The keyword arguments of the Python `relate`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelateKwargs {
    #[serde(with = "OnRowErrorDef")]
    on_row_error: OnRowError,
    #[serde(with = "EarlyCoordCheckDef")]
    early_coord_check: EarlyCoordCheck,
}

fn kernel_output_type_relate(input_fields: &[Field], _kwargs: RelateKwargs) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "relate takes two geometries, got {} inputs", input_fields.len());
    };
    ensure_same_crs(&describe(a.dtype())?, &describe(b.dtype())?)?;
    Ok(Field::new(a.name().clone(), DataType::String))
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_relate)]
fn relate(inputs: &[Series], kwargs: RelateKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let (ignore, early) = (kwargs.on_row_error, kwargs.early_coord_check);
    let chunk_pairs = match single_chunk_masked_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs,
        // `RelatePairs` pairs a single row with every row itself,
        // so the single geometry stays on its own side.
        GeometryPairs::OneAgainstMany { a, b } => b.into_iter().map(|b| (a.clone(), b)).collect(),
        GeometryPairs::ManyAgainstOne { a, b } => a.into_iter().map(|a| (a, b.clone())).collect(),
        GeometryPairs::AllNull(lengths) => {
            let chunks = lengths
                .iter()
                .map(|&rows| Utf8ViewArray::new_null(ArrowDataType::Utf8View, rows));
            return Ok(StringChunked::from_chunk_iter(a.name().clone(), chunks).into_series());
        }
    };
    let (a_kind, b_kind) = (kind_of(a)?, kind_of(b)?);
    let chunks = chunk_pairs
        .iter()
        .map(|(a, b)| geopolars_core::relate(a.as_ref(), a_kind, b.as_ref(), b_kind, ignore, early))
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(StringChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
