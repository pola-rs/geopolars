//! Whether two geometries are topologically equal, forwarded to GEOS.
//! Implements the Python `is_topologically_equal`.

use crate::common_arguments::{EarlyCoordCheckDef, OnRowErrorDef};
use geopolars_compute::{EarlyCoordCheck, OnRowError};
use geopolars_core::are_topologically_equal;
use polars::prelude::*;
use polars_arrow::array::BooleanArray;
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{
    describe, ensure_same_crs, kind_of, single_chunk_pairs, GeometryPairs,
};

/// The keyword arguments of the Python `is_topologically_equal`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsTopologicallyEqualKwargs {
    #[serde(with = "OnRowErrorDef")]
    on_row_error: OnRowError,
    #[serde(with = "EarlyCoordCheckDef")]
    early_coord_check: EarlyCoordCheck,
}

fn kernel_output_type_is_topologically_equal(
    input_fields: &[Field],
    _kwargs: IsTopologicallyEqualKwargs,
) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "is_topologically_equal takes two geometries, got {} inputs", input_fields.len());
    };
    ensure_same_crs(&describe(a.dtype())?, &describe(b.dtype())?)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_is_topologically_equal)]
fn is_topologically_equal(
    inputs: &[Series],
    kwargs: IsTopologicallyEqualKwargs,
) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let (on_row_error, early) = (kwargs.on_row_error, kwargs.early_coord_check);
    let chunk_pairs = match single_chunk_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs,
        GeometryPairs::OneAgainstMany { a, b } => b.into_iter().map(|b| (a.clone(), b)).collect(),
        GeometryPairs::ManyAgainstOne { a, b } => a.into_iter().map(|a| (a, b.clone())).collect(),
        GeometryPairs::AllNull(lengths) => {
            let chunks = lengths
                .iter()
                .map(|&rows| BooleanArray::new_null(ArrowDataType::Boolean, rows));
            return Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series());
        }
    };
    let (a_kind, b_kind) = (kind_of(a)?, kind_of(b)?);
    let chunks = chunk_pairs
        .iter()
        .map(|(a, b)| {
            are_topologically_equal(a.as_ref(), a_kind, b.as_ref(), b_kind, on_row_error, early)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
