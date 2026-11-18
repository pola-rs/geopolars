//! The shortest line between two geometries.
//! Implements the Python `shortest_line_between`.

use crate::common_arguments::{EarlyCoordCheckDef, OnRowErrorDef};
use geopolars_compute::{EarlyCoordCheck, OnRowError};
use geopolars_core::shortest_lines;
use geopolars_views::{GeoDimension, Kind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{
    describe, ensure_same_crs, geometry_series, kind_of, single_chunk_masked_pairs, Geo,
    GeometryPairs,
};

/// The keyword arguments of the Python `shortest_line_between`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortestLineBetweenKwargs {
    #[serde(with = "OnRowErrorDef")]
    on_row_error: OnRowError,
    #[serde(with = "EarlyCoordCheckDef")]
    early_coord_check: EarlyCoordCheck,
}

fn shortest_line_between_type(a: &DataType, b: &DataType) -> PolarsResult<Geo> {
    let (a, b) = (describe(a)?, describe(b)?);
    ensure_same_crs(&a, &b)?;
    Ok(Geo::new(
        Kind::LineString,
        GeoDimension::XY,
        a.metadata.clone(),
    ))
}

fn kernel_output_type_shortest_line_between(
    input_fields: &[Field],
    _kwargs: ShortestLineBetweenKwargs,
) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "shortest_line_between takes two geometries, got {} inputs", input_fields.len());
    };
    Ok(Field::new(
        a.name().clone(),
        shortest_line_between_type(a.dtype(), b.dtype())?.dtype(),
    ))
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_shortest_line_between)]
fn shortest_line_between(
    inputs: &[Series],
    kwargs: ShortestLineBetweenKwargs,
) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let lines = shortest_line_between_type(a.dtype(), b.dtype())?;
    let name = a.name().clone();
    let chunk_pairs = match single_chunk_masked_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs,
        GeometryPairs::OneAgainstMany { a, b } => b.into_iter().map(|b| (a.clone(), b)).collect(),
        GeometryPairs::ManyAgainstOne { a, b } => a.into_iter().map(|a| (a, b.clone())).collect(),
        GeometryPairs::AllNull(lengths) => {
            return Ok(Series::full_null(
                name,
                lengths.iter().sum(),
                &lines.dtype(),
            ));
        }
    };
    let (a_kind, b_kind) = (kind_of(a)?, kind_of(b)?);
    let (on_row_error, early) = (kwargs.on_row_error, kwargs.early_coord_check);
    let chunks = chunk_pairs
        .iter()
        .map(|(a, b)| shortest_lines(a.as_ref(), a_kind, b.as_ref(), b_kind, on_row_error, early))
        .collect::<PolarsResult<Vec<_>>>()?;
    geometry_series(name, lines, chunks)
}
