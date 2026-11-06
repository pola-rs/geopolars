//! The shortest line between two geometries
//! (see [`geopolars_compute::geos::shortest_lines_pairwise`]).

use geopolars_compute::geos::{shortest_lines_against_one, shortest_lines_pairwise};
use geopolars_core::GeoDimension;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{
    describe, ensure_same_crs, geometry_array, kind_of, storage_as_single_chunk, Geo, Kind,
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

/// Runs [shortest_lines_against_one] on every chunk of `many`.
fn every_shortest_line_against_one(
    one: &Series,
    many: &Series,
    lines_start_on_one: bool,
    kwargs: &ShortestLineBetweenKwargs,
) -> PolarsResult<Vec<ArrayRef>> {
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let one = geometry_array(storage_as_single_chunk(one)?.as_ref(), kind_of(one)?)?;
    let kind = kind_of(many)?;
    many.ext()?
        .storage()
        .chunks()
        .iter()
        .map(|many| {
            let many = geometry_array(many.as_ref(), kind)?;
            let lines = shortest_lines_against_one(&one, &many, lines_start_on_one, ignore, early)
                .into_polars()?;
            Ok(lines.into_list_array().boxed())
        })
        .collect()
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
    let chunks = match (a.len(), b.len()) {
        // The kinds can differ, but `aligned_chunks` needs two storages of one physical type.
        // So both sides are made a single chunk instead.
        (n, m) if n == m => {
            let a = geometry_array(storage_as_single_chunk(a)?.as_ref(), kind_of(a)?)?;
            let b = geometry_array(storage_as_single_chunk(b)?.as_ref(), kind_of(b)?)?;
            let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
            let lines = shortest_lines_pairwise(a, b, ignore, early).into_polars()?;
            vec![lines.into_list_array().boxed()]
        }
        (1, _) => every_shortest_line_against_one(a, b, true, &kwargs)?,
        (_, 1) => every_shortest_line_against_one(b, a, false, &kwargs)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot pair {n} geometries with {m}"),
    };
    // SAFETY: the shortest lines are XY linestrings, as `dtype` says.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(name, chunks, &dtype) })
}
