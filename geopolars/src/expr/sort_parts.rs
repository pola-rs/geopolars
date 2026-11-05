//! Sorting the parts of a geometry.
//! See `sort_parts` in Python for the rules.

use geopolars_compute::native::{
    multi_line_strings_sorted, multi_points_sorted, multi_polygons_sorted,
    polygons_with_holes_sorted,
};
use geopolars_core::{
    downcast, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::{describe, Kind};

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortPartsKwargs {
    descending: bool,
}

/// The kernel of `kind` on one chunk of the storage.
fn chunk_sorted(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &SortPartsKwargs,
) -> PolarsResult<Box<dyn Array>> {
    let descending = kwargs.descending;
    Ok(match kind {
        // Python hands these back unchanged, without calling the kernel.
        Kind::Point | Kind::LineString => {
            polars_bail!(InvalidOperation: "sort_parts has no kernel for {kind:?}")
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            multi_points_sorted(&array, descending)
                .into_list_array()
                .boxed()
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            polygons_with_holes_sorted(&array, descending)
                .into_list_array()
                .boxed()
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            multi_line_strings_sorted(&array, descending)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            multi_polygons_sorted(&array, descending)
                .into_list_array()
                .boxed()
        }
    })
}

/// See `sort_parts`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn sort_parts(inputs: &[Series], kwargs: SortPartsKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_sorted(chunk.as_ref(), kind, &kwargs))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
