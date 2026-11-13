//! Fixing the coordinates outside the CRS's bounds.
//! Implements the Python `fix_out_of_bounds`.

use geopolars_compute::native::{Bounds, CoordinateSelection, OutOfBoundsAction, TooShort};
use geopolars_core::{with_clamped_coords, with_selected_coords};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use super::remove_repeats_in_succession::TooShortDef;
use crate::geoarrow::{describe, geometry_series};

#[derive(Deserialize)]
#[serde(remote = "OutOfBoundsAction", rename_all = "snake_case")]
enum OutOfBoundsActionDef {
    SetEmpty,
    Drop,
    SetNull,
    Clamp,
}

/// The keyword arguments of the Python `fix_out_of_bounds`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixOutOfBoundsKwargs {
    /// The bounds that `crs_bounds` names, looked up at plan time.
    bounds: Bounds,
    #[serde(with = "OutOfBoundsActionDef")]
    action: OutOfBoundsAction,
    #[serde(with = "TooShortDef")]
    too_short: TooShort,
}

#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn fix_out_of_bounds(inputs: &[Series], kwargs: FixOutOfBoundsKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let FixOutOfBoundsKwargs {
        bounds,
        action,
        too_short,
    } = kwargs;
    let selection = CoordinateSelection {
        non_finite: None,
        bounds: Some(bounds),
    };
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| match action.selected_action() {
            Some(action) => {
                with_selected_coords(chunk.as_ref(), kind, &selection, action, too_short)
            }
            None => with_clamped_coords(chunk.as_ref(), kind, &bounds),
        })
        .collect::<PolarsResult<_>>()?;
    let geo = describe(inputs[0].dtype())?.geo();
    geometry_series(inputs[0].name().clone(), geo, chunks)
}
