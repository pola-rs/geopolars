//! The shortest line between two geometries
//! (see [`geopolars_compute::geos::shortest_lines_pairwise`]).

use geopolars_compute::geos::{shortest_lines_against_one, shortest_lines_pairwise};
use geopolars_core::GeoDimension;
use polars::prelude::*;
use polars_arrow::compute::utils::combine_validities_and;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{
    describe, ensure_same_crs, geometry_array, kind_of, storage_as_single_chunk, Geo, Kind,
};

/// See python
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShortestLineKwargs {
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// An XY linestring with the metadata of the first geometry.
/// Both can be of any kind and dimension, in the same CRS (or neither declaring one).
fn shortest_line_type(a: &DataType, b: &DataType) -> PolarsResult<DataType> {
    let (a, b) = (describe(a)?, describe(b)?);
    ensure_same_crs(&a, &b)?;
    Ok(Geo::new(Kind::LineString, GeoDimension::XY, a.metadata.clone()).dtype())
}

/// For [`shortest_line`], named after the first geometry.
fn kernel_output_type_shortest_line(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "shortest_line takes two geometries, got {} inputs", input_fields.len());
    };
    Ok(Field::new(
        a.name().clone(),
        shortest_line_type(a.dtype(), b.dtype())?,
    ))
}

/// Runs [shortest_lines_against_one] on every chunk of `many`, against the single row of `one`.
fn every_shortest_line_to_one(
    one: &Series,
    many: &Series,
    one_is_a: bool,
    kwargs: &ShortestLineKwargs,
) -> PolarsResult<Option<Vec<ArrayRef>>> {
    let one_chunk = storage_as_single_chunk(one)?;
    // A null `one` makes every result null, so `many` is not sent to GEOS, where a row could fail.
    if one_chunk.is_null(0) {
        return Ok(None);
    }
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let one = geometry_array(one_chunk.as_ref(), kind_of(one)?)?;
    let kind = kind_of(many)?;
    let chunks = many
        .ext()?
        .storage()
        .chunks()
        .iter()
        .map(|many| {
            let many = geometry_array(many.as_ref(), kind)?;
            let lines =
                shortest_lines_against_one(&one, &many, one_is_a, ignore, early).into_polars()?;
            Ok(lines.into_list_array().boxed())
        })
        .collect::<PolarsResult<_>>()?;
    Ok(Some(chunks))
}

/// See `shortest_line`.
#[polars_expr(output_type_func=kernel_output_type_shortest_line)]
fn shortest_line(inputs: &[Series], kwargs: ShortestLineKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let dtype = shortest_line_type(a.dtype(), b.dtype())?;
    let name = a.name().clone();
    let chunks = match (a.len(), b.len()) {
        // The kinds can differ, but `aligned_chunks` needs two storages of one physical type.
        // So both sides are made a single chunk instead.
        (n, m) if n == m => {
            let (a_chunk, b_chunk) = (storage_as_single_chunk(a)?, storage_as_single_chunk(b)?);
            // A row null on one side is made null on the other too,
            // so that it is not sent to GEOS, where it could fail.
            let validity = combine_validities_and(a_chunk.validity(), b_chunk.validity());
            let (a, b) = (
                geometry_array(
                    a_chunk.with_validity(validity.clone()).as_ref(),
                    kind_of(a)?,
                )?,
                geometry_array(b_chunk.with_validity(validity).as_ref(), kind_of(b)?)?,
            );
            let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
            let lines = shortest_lines_pairwise(&a, &b, ignore, early).into_polars()?;
            Some(vec![lines.into_list_array().boxed()])
        }
        (1, _) => every_shortest_line_to_one(a, b, true, &kwargs)?,
        (_, 1) => every_shortest_line_to_one(b, a, false, &kwargs)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot pair {n} geometries with {m}"),
    };
    let Some(chunks) = chunks else {
        return Ok(Series::full_null(name, a.len().max(b.len()), &dtype));
    };
    // SAFETY: the shortest lines are XY linestrings, as `dtype` says.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(name, chunks, &dtype) })
}
