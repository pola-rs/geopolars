//! Emptying, dropping or nulling the coordinates that are not finite, or empty.
//! Implements the Python `filter_coordinates`.

use geopolars_compute::native::{CoordinateSelection, NonFinite, SelectedAction};
use geopolars_compute::ShortLines;
use geopolars_core::with_selected_coords;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_wrapper_same_geometry;
use crate::common_arguments::ShortLinesDef;
use crate::geoarrow::{describe, geometry_series};

#[derive(Deserialize)]
#[serde(remote = "SelectedAction", rename_all = "snake_case")]
enum SelectedActionDef {
    SetEmpty,
    Drop,
    SetNull,
}

/// The keyword arguments of the Python `filter_coordinates`, as the Python `NonFinite` holds them.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonFiniteKwargs {
    pub(super) select_non_finite: bool,
    pub(super) select_empty: bool,
    pub(super) ignore_z: bool,
    pub(super) ignore_m: bool,
    #[serde(with = "SelectedActionDef")]
    pub(super) action: SelectedAction,
    #[serde(with = "ShortLinesDef")]
    pub(super) short_lines: ShortLines,
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

fn kernel_output_type_filter_coordinates(
    input_fields: &[Field],
    _kwargs: NonFiniteKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_same_geometry(input_fields)
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_filter_coordinates)]
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
                kwargs.short_lines,
            )
        })
        .collect::<PolarsResult<_>>()?;
    let geo = describe(inputs[0].dtype())?.geo();
    geometry_series(inputs[0].name().clone(), geo, chunks)
}
