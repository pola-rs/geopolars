//! Emptying, dropping or nulling the coordinates that are not finite, or empty,
//! through `geopolars_compute::native::*_with_selected`.
//! See `filter_coordinates` in Python for the rules.

use geopolars_compute::native::{
    line_strings_with_selected, multi_line_strings_with_selected, multi_points_with_selected,
    multi_polygons_with_selected, points_with_selected, polygons_with_selected,
    CoordinateSelection, NonFinite, SelectedAction, TooShort,
};
use geopolars_core::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use super::remove_repeats_in_succession::TooShortDef;
use crate::geoarrow::describe;

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

/// `action` on the coordinates `selection` selects, for one chunk of the storage of a `kind`.
/// Shared by [`filter_coordinates`] and `fix_out_of_bounds`.
pub(super) fn chunk_with_selected(
    chunk: &dyn Array,
    kind: Kind,
    selection: &CoordinateSelection,
    action: SelectedAction,
    too_short: TooShort,
) -> PolarsResult<Box<dyn Array>> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => {
            points_with_selected(&PointArray::new(Coords::new(chunk)?), selection, action)
                .into_struct_array()
                .boxed()
        }
        Kind::LineString => line_strings_with_selected(
            &LineStringArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiPoint => multi_points_with_selected(
            &MultiPointArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )
        .into_list_array()
        .boxed(),
        Kind::Polygon => {
            polygons_with_selected(&PolygonArray::new(rows()?)?, selection, action, too_short)
                .into_list_array()
                .boxed()
        }
        Kind::MultiLineString => multi_line_strings_with_selected(
            &MultiLineStringArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiPolygon => multi_polygons_with_selected(
            &MultiPolygonArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )
        .into_list_array()
        .boxed(),
    })
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
            chunk_with_selected(
                chunk.as_ref(),
                kind,
                &selection,
                kwargs.action,
                kwargs.too_short,
            )
        })
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
