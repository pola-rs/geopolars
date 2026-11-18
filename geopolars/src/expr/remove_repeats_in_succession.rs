//! Removing the points that coincide with the one before them.
//! Implements the Python `remove_repeats_in_succession`.

use geopolars_compute::native::{Coincidence, CollapsedLines, LineOptions};
use geopolars_compute::ShortLines;
use geopolars_core::without_repeats_in_succession;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_wrapper_same_geometry;
use crate::common_arguments::ShortLinesDef;
use crate::geoarrow::{describe, geometry_series};

#[derive(Deserialize)]
#[serde(remote = "CollapsedLines", rename_all = "snake_case")]
enum CollapsedLinesDef {
    KeepEnds,
    SingleVertex,
}

/// The keyword arguments of the Python `remove_repeats_in_succession`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoveRepeatsInSuccessionKwargs {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    unique_z: bool,
    unique_m: bool,
    #[serde(with = "CollapsedLinesDef")]
    collapsed_lines: CollapsedLines,
    #[serde(with = "ShortLinesDef")]
    short_lines: ShortLines,
}

fn kernel_output_type_remove_repeats_in_succession(
    input_fields: &[Field],
    _kwargs: RemoveRepeatsInSuccessionKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_same_geometry(input_fields)
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_remove_repeats_in_succession)]
fn remove_repeats_in_succession(
    inputs: &[Series],
    kwargs: RemoveRepeatsInSuccessionKwargs,
) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let coincidence = Coincidence::new(kwargs.abs_tol, kwargs.rel_tol);
    let options = LineOptions {
        collapsed_lines: kwargs.collapsed_lines,
        short_lines: kwargs.short_lines,
    };
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            without_repeats_in_succession(
                chunk.as_ref(),
                kind,
                coincidence,
                kwargs.unique_z,
                kwargs.unique_m,
                options,
            )
        })
        .collect::<PolarsResult<_>>()?;
    let geo = describe(inputs[0].dtype())?.geo();
    geometry_series(inputs[0].name().clone(), geo, chunks)
}
