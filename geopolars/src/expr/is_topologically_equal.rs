//! Whether two geometries are topologically equal
//! (see [`geopolars_compute::geos::topologically_equal_pairwise`]).

use geopolars_compute::geos::{topologically_equal_against_one, topologically_equal_pairwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{describe, ensure_same_crs, map_geometry_pairs};

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
    let chunks = map_geometry_pairs(
        a,
        b,
        |a, b| topologically_equal_pairwise(a, b, ignore, early).into_polars(),
        |one, many, _| topologically_equal_against_one(one, many, ignore, early).into_polars(),
    )?;
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
