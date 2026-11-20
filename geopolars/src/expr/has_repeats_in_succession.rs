//! Whether a geometry has two points in a row that coincide.
//! Implements the Python `has_repeats_in_succession`.

use geopolars_compute::native::Coincidence;
use geopolars_core::have_repeats_in_succession;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::describe;

/// The keyword arguments of the Python `has_repeats_in_succession`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HasRepeatsInSuccessionKwargs {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    include_z: bool,
    include_m: bool,
}

fn kernel_output_type_has_repeats_in_succession(
    input_fields: &[Field],
    _kwargs: HasRepeatsInSuccessionKwargs,
) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_has_repeats_in_succession)]
fn has_repeats_in_succession(
    inputs: &[Series],
    kwargs: HasRepeatsInSuccessionKwargs,
) -> PolarsResult<Series> {
    let geometries = &inputs[0];
    let kind = describe(geometries.dtype())?.kind;
    let coincidence = Coincidence::new(kwargs.abs_tol, kwargs.rel_tol);
    let storage = geometries.ext()?.storage();
    let chunks = storage.chunks().iter().map(|chunk| {
        have_repeats_in_succession(
            chunk.as_ref(),
            kind,
            coincidence,
            kwargs.include_z,
            kwargs.include_m,
        )
    });
    Ok(BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)?.into_series())
}
