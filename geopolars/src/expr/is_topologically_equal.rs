//! Whether two geometries are topologically equal
//! (see [`geopolars_compute::geos::topologically_equal_pairwise`]).

use geopolars_compute::geos::{topologically_equal_against_one, topologically_equal_pairwise};
use polars::prelude::*;
use polars_arrow::array::BooleanArray;
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{
    describe, ensure_same_crs, geometry_array, kind_of, single_chunk_masked_pairs, GeometryPairs,
    Kind,
};

/// See python
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsTopologicallyEqualKwargs {
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// For [`is_topologically_equal`]: a `bool`, named after the first geometry.
///
/// Both can be of any kind and dimension, in the same CRS (or neither declaring one).
fn kernel_output_type_is_topologically_equal(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "is_topologically_equal takes two geometries, got {} inputs", input_fields.len());
    };
    ensure_same_crs(&describe(a.dtype())?, &describe(b.dtype())?)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

/// See `is_topologically_equal`.
#[polars_expr(output_type_func=kernel_output_type_is_topologically_equal)]
fn is_topologically_equal(
    inputs: &[Series],
    kwargs: IsTopologicallyEqualKwargs,
) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let (a_kind, b_kind) = (kind_of(a)?, kind_of(b)?);
    let chunks = match single_chunk_masked_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs
            .iter()
            .map(|(a, b)| {
                let (a, b) = (
                    geometry_array(a.as_ref(), a_kind)?,
                    geometry_array(b.as_ref(), b_kind)?,
                );
                topologically_equal_pairwise(&a, &b, ignore, early).into_polars()
            })
            .collect::<PolarsResult<Vec<_>>>()?,
        GeometryPairs::OneAgainstMany { a, b } => {
            is_topologically_equal_against_one(&a, a_kind, &b, b_kind, ignore, early)?
        }
        GeometryPairs::ManyAgainstOne { a, b } => {
            is_topologically_equal_against_one(&b, b_kind, &a, a_kind, ignore, early)?
        }
        GeometryPairs::AllNull(lengths) => lengths
            .iter()
            .map(|&rows| BooleanArray::new_null(ArrowDataType::Boolean, rows))
            .collect(),
    };
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}

fn is_topologically_equal_against_one(
    one: &ArrayRef,
    one_kind: Kind,
    many: &[ArrayRef],
    many_kind: Kind,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> PolarsResult<Vec<BooleanArray>> {
    let one = geometry_array(one.as_ref(), one_kind)?;
    many.iter()
        .map(|many| {
            let many = geometry_array(many.as_ref(), many_kind)?;
            topologically_equal_against_one(&one, &many, ignore_per_row_errors, early_finite_check)
                .into_polars()
        })
        .collect()
}
