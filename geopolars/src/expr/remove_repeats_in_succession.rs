//! Removing the points that coincide with the one before them,
//! through [`geopolars_core::without_repeats_in_succession`].
//! See `remove_repeats_in_succession` in Python for the rules.

use geopolars_compute::native::{Coincidence, CollapsedLines, LineOptions, TooShort};
use geopolars_core::without_repeats_in_succession;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::{describe, same_geometry_series};

#[derive(Deserialize)]
#[serde(remote = "CollapsedLines", rename_all = "snake_case")]
enum CollapsedLinesDef {
    KeepEnds,
    SingleVertex,
}

#[derive(Deserialize)]
#[serde(remote = "TooShort", rename_all = "snake_case")]
pub(super) enum TooShortDef {
    Null,
    Keep,
    Empty,
}

/// See python
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoveRepeatsInSuccessionKwargs {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    unique_z: bool,
    unique_m: bool,
    #[serde(with = "CollapsedLinesDef")]
    collapsed_lines: CollapsedLines,
    #[serde(with = "TooShortDef")]
    too_short: TooShort,
}

/// See `remove_repeats_in_succession`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn remove_repeats_in_succession(
    inputs: &[Series],
    kwargs: RemoveRepeatsInSuccessionKwargs,
) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let coincidence = Coincidence::new(kwargs.abs_tol, kwargs.rel_tol);
    let options = LineOptions {
        collapsed_lines: kwargs.collapsed_lines,
        too_short: kwargs.too_short,
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
    Ok(same_geometry_series(inputs[0].name().clone(), geo, chunks))
}
