//! The shortest line between two geometries
//! (see [`geopolars_compute::geos::shortest_lines_pairwise`]).

use geopolars_compute::geos::{shortest_lines_against_one, shortest_lines_pairwise};
use geopolars_compute::GeoComputeResult;
use geopolars_core::{GeoDimension, LineStringArray};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{describe, ensure_same_crs, pair_geometries, Geo, Kind};

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
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let chunks = pair_geometries(
        a,
        b,
        |a, b| lines_chunk(shortest_lines_pairwise(a, b, ignore, early)),
        |one, many, one_is_a| {
            lines_chunk(shortest_lines_against_one(
                one, many, one_is_a, ignore, early,
            ))
        },
    )?;
    // SAFETY: the shortest lines are XY linestrings, as `dtype` says.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(name, chunks, &dtype) })
}

fn lines_chunk(lines: GeoComputeResult<LineStringArray>) -> PolarsResult<ArrayRef> {
    Ok(lines.into_polars()?.into_list_array().boxed())
}
