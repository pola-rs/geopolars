//! The shortest line between two geometries
//! (see [`geopolars_compute::geos::shortest_lines_pairwise`]).

use geopolars_core::{pairwise_shortest_lines, shortest_lines_to_one};
use geopolars_views::{GeoDimension, Kind, LineStringArray};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{
    describe, ensure_same_crs, kind_of, single_chunk_masked_pairs, Geo, GeometryPairs,
};

/// See python
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShortestLineBetweenKwargs {
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

fn shortest_line_between_type(a: &DataType, b: &DataType) -> PolarsResult<DataType> {
    let (a, b) = (describe(a)?, describe(b)?);
    ensure_same_crs(&a, &b)?;
    Ok(Geo::new(Kind::LineString, GeoDimension::XY, a.metadata.clone()).dtype())
}

/// For [`shortest_line_between`], named after the first geometry.
fn kernel_output_type_shortest_line_between(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "shortest_line_between takes two geometries, got {} inputs", input_fields.len());
    };
    Ok(Field::new(
        a.name().clone(),
        shortest_line_between_type(a.dtype(), b.dtype())?,
    ))
}

/// See `shortest_line_between`.
#[polars_expr(output_type_func=kernel_output_type_shortest_line_between)]
fn shortest_line_between(
    inputs: &[Series],
    kwargs: ShortestLineBetweenKwargs,
) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let dtype = shortest_line_between_type(a.dtype(), b.dtype())?;
    let name = a.name().clone();
    let (a_kind, b_kind) = (kind_of(a)?, kind_of(b)?);
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let against_one =
        |one: ArrayRef, one_kind: Kind, many: Vec<ArrayRef>, many_kind: Kind, one_is_a| {
            many.iter()
                .map(|many| {
                    lines_chunk(shortest_lines_to_one(
                        one.as_ref(),
                        one_kind,
                        many.as_ref(),
                        many_kind,
                        one_is_a,
                        ignore,
                        early,
                    ))
                })
                .collect::<PolarsResult<Vec<_>>>()
        };
    let chunks = match single_chunk_masked_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs
            .iter()
            .map(|(a, b)| {
                lines_chunk(pairwise_shortest_lines(
                    a.as_ref(),
                    a_kind,
                    b.as_ref(),
                    b_kind,
                    ignore,
                    early,
                ))
            })
            .collect::<PolarsResult<Vec<_>>>()?,
        GeometryPairs::OneAgainstMany { a: one, b: many } => {
            against_one(one, a_kind, many, b_kind, true)?
        }
        GeometryPairs::ManyAgainstOne { a: many, b: one } => {
            against_one(one, b_kind, many, a_kind, false)?
        }
        GeometryPairs::AllNull(lengths) => {
            return Ok(Series::full_null(name, lengths.iter().sum(), &dtype));
        }
    };
    // SAFETY: the shortest lines are XY linestrings, as `dtype` says.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(name, chunks, &dtype) })
}

fn lines_chunk(lines: PolarsResult<LineStringArray>) -> PolarsResult<ArrayRef> {
    Ok(lines?.into_list_array().boxed())
}
