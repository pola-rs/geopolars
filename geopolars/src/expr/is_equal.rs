//! Whether two geometries are equal, coordinate by coordinate

use geopolars_compute::native::Coincidence;
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray};
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{aligned_pairs, describe, ensure_same_crs, GeometryPairs};

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsEqualKwargs {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    nans_equal: bool,
    unique_z: bool,
    unique_m: bool,
}

/// For [`is_equal`]: a `bool`, named after the first geometry.
///
/// Both have to be the same kind and dimension, in the same CRS (or neither declaring one).
fn kernel_output_type_is_equal(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "is_equal takes two geometries, got {} inputs", input_fields.len());
    };
    let (geo_a, geo_b) = (describe(a.dtype())?, describe(b.dtype())?);
    polars_ensure!(
        geo_a.kind == geo_b.kind && geo_a.dim == geo_b.dim,
        SchemaMismatch: "is_equal expects two geometries of the same type, got: {} and {}",
        a.dtype(), b.dtype()
    );
    ensure_same_crs(&geo_a, &geo_b)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

/// See `is_equal`.
#[polars_expr(output_type_func=kernel_output_type_is_equal)]
fn is_equal(inputs: &[Series], kwargs: IsEqualKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let kind = describe(a.dtype())?.kind;
    let coincidence = Coincidence::new(kwargs.abs_tol, kwargs.rel_tol);
    let are_equal = |a: &dyn Array, b: &dyn Array| {
        geopolars_core::are_equal(
            a,
            b,
            kind,
            coincidence,
            kwargs.unique_z,
            kwargs.unique_m,
            kwargs.nans_equal,
        )
    };
    let chunks = match aligned_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs
            .iter()
            .map(|(a, b)| are_equal(a.as_ref(), b.as_ref()))
            .collect::<PolarsResult<Vec<_>>>()?,
        // Equality goes both ways, so it does not matter which side `one` came from.
        GeometryPairs::OneAgainstMany { a: one, b: many }
        | GeometryPairs::ManyAgainstOne { a: many, b: one } => many
            .iter()
            .map(|many| are_equal(many.as_ref(), one.as_ref()))
            .collect::<PolarsResult<Vec<_>>>()?,
        GeometryPairs::AllNull(lengths) => lengths
            .iter()
            .map(|&rows| BooleanArray::new_null(ArrowDataType::Boolean, rows))
            .collect(),
    };
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
