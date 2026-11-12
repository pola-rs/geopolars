//! Removing the points that coincide with the one before them,
//! through `geopolars_compute::native::*_without_repeats_in_succession`.
//! See `remove_repeats_in_succession` in Python for the rules.

use geopolars_compute::native::{
    line_strings_without_repeats_in_succession, multi_line_strings_without_repeats_in_succession,
    multi_points_without_repeats_in_succession, multi_polygons_without_repeats_in_succession,
    polygons_without_repeats_in_succession, Coincidence, CollapsedLines, LineOptions, TooShort,
};
use geopolars_views::{
    downcast, Kind, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::describe;

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

/// [`remove_repeats_in_succession`] for one chunk of the storage of a `kind`.
fn chunk_without_repeats_in_succession(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &RemoveRepeatsInSuccessionKwargs,
) -> PolarsResult<Box<dyn Array>> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    let coincidence = Coincidence::new(kwargs.abs_tol, kwargs.rel_tol);
    let (unique_z, unique_m, too_short) = (kwargs.unique_z, kwargs.unique_m, kwargs.too_short);
    let options = LineOptions {
        collapsed_lines: kwargs.collapsed_lines,
        too_short,
    };
    Ok(match kind {
        Kind::Point => chunk.to_boxed(),
        Kind::LineString => line_strings_without_repeats_in_succession(
            &LineStringArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
            options,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiPoint => multi_points_without_repeats_in_succession(
            &MultiPointArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        )
        .into_list_array()
        .boxed(),
        Kind::Polygon => polygons_without_repeats_in_succession(
            &PolygonArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
            too_short,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiLineString => multi_line_strings_without_repeats_in_succession(
            &MultiLineStringArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
            options,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiPolygon => multi_polygons_without_repeats_in_succession(
            &MultiPolygonArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
            too_short,
        )
        .into_list_array()
        .boxed(),
    })
}

/// See `remove_repeats_in_succession`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn remove_repeats_in_succession(
    inputs: &[Series],
    kwargs: RemoveRepeatsInSuccessionKwargs,
) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_without_repeats_in_succession(chunk.as_ref(), kind, &kwargs))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
