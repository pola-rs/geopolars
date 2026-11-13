//! Snapping every coordinate onto a grid, through `geopolars_compute::native::*_snapped_to_grid`.
//! See `snap_to_grid` in Python for the rules.

use geopolars_compute::native::{Grid, RoundMode};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::{describe, same_geometry_series};

#[derive(Deserialize)]
#[serde(remote = "RoundMode", rename_all = "snake_case")]
enum RoundModeDef {
    HalfToEven,
    HalfAwayFromZero,
    ToZero,
}

/// See [`Grid`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapToGridKwargs {
    spacing: [f64; 3],
    origin: [f64; 3],
    #[serde(with = "RoundModeDef")]
    mode: RoundMode,
}

/// See `snap_to_grid`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn snap_to_grid(inputs: &[Series], kwargs: SnapToGridKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let SnapToGridKwargs {
        spacing,
        origin,
        mode,
    } = kwargs;
    let grid = Grid {
        spacing,
        origin,
        mode,
    };
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| geopolars_core::snap_to_grid(chunk.as_ref(), kind, &grid))
        .collect::<PolarsResult<_>>()?;
    let geo = describe(inputs[0].dtype())?.geo();
    Ok(same_geometry_series(inputs[0].name().clone(), geo, chunks))
}
