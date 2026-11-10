//! Predicates on two geometries, such as whether they intersect
//! (see [`geopolars_compute::geos::test_predicate`]).

use geopolars_compute::geos::{test_predicate, Predicate};
use polars::prelude::*;
use polars_arrow::array::BooleanArray;
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{
    describe, ensure_same_crs, geometry_array, kind_of, single_chunk_masked_pairs, GeometryPairs,
};

/// See python
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PredicateKwargs {
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// For every predicate: a `bool`, named after the first geometry.
///
/// Both can be of any kind and dimension, in the same CRS (or neither declaring one).
fn kernel_output_type_predicate(name: &str, input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "{name} takes two geometries, got {} inputs", input_fields.len());
    };
    ensure_same_crs(&describe(a.dtype())?, &describe(b.dtype())?)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

fn predicate_series(
    inputs: &[Series],
    kwargs: PredicateKwargs,
    predicate: Predicate,
) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let chunk_pairs = match single_chunk_masked_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs,
        // `test_predicate` pairs a single row with every row itself,
        // so the single geometry stays on its own side.
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
            let (a, b) = (
                geometry_array(a.as_ref(), a_kind)?,
                geometry_array(b.as_ref(), b_kind)?,
            );
            test_predicate(&a, &b, predicate, ignore, early).into_polars()
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}

fn kernel_output_type_intersects(input_fields: &[Field]) -> PolarsResult<Field> {
    kernel_output_type_predicate("intersects", input_fields)
}

/// See `intersects`.
#[polars_expr(output_type_func=kernel_output_type_intersects)]
fn intersects(inputs: &[Series], kwargs: PredicateKwargs) -> PolarsResult<Series> {
    predicate_series(inputs, kwargs, Predicate::Intersects)
}
