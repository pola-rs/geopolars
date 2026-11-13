//! Emptying, dropping or nulling the coordinates that are not finite, or empty,
//! through [`geopolars_core::with_selected_coords`].
//! See `filter_coordinates` in Python for the rules.

use geopolars_compute::native::{CoordinateSelection, NonFinite, SelectedAction, TooShort};
use geopolars_core::with_selected_coords;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use super::remove_repeats_in_succession::TooShortDef;
use crate::geoarrow::{describe, geometry_series};

#[derive(Deserialize)]
#[serde(remote = "SelectedAction", rename_all = "snake_case")]
enum SelectedActionDef {
    SetEmpty,
    Drop,
    SetNull,
}

/// See python
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonFiniteKwargs {
    pub(super) select_non_finite: bool,
    pub(super) select_empty: bool,
    pub(super) ignore_z: bool,
    pub(super) ignore_m: bool,
    #[serde(with = "SelectedActionDef")]
    pub(super) action: SelectedAction,
    #[serde(with = "TooShortDef")]
    pub(super) too_short: TooShort,
}

impl NonFiniteKwargs {
    fn selection(&self) -> CoordinateSelection {
        CoordinateSelection {
            non_finite: Some(NonFinite {
                select_non_finite: self.select_non_finite,
                select_empty: self.select_empty,
                ignore_z: self.ignore_z,
                ignore_m: self.ignore_m,
            }),
            bounds: None,
        }
    }
}

/// See `filter_coordinates`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn filter_coordinates(inputs: &[Series], kwargs: NonFiniteKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let selection = kwargs.selection();
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            with_selected_coords(
                chunk.as_ref(),
                kind,
                &selection,
                kwargs.action,
                kwargs.too_short,
            )
        })
        .collect::<PolarsResult<_>>()?;
    let geo = describe(inputs[0].dtype())?.geo();
    geometry_series(inputs[0].name().clone(), geo, chunks)
}
