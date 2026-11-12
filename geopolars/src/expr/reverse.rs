//! Reversing the order of a geometry's vertices.
//! See `reverse` in Python for the rules.

use geopolars_compute::native::{
    reverse_line_strings, reverse_multi_line_strings, reverse_multi_points, reverse_multi_polygons,
    reverse_polygons,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, Kind, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::describe;

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReverseKwargs {
    reverse_holes: bool,
    nested: bool,
}

/// The kernel of `kind` on one chunk of the storage.
fn chunk_reversed(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &ReverseKwargs,
) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        Kind::Point => polars_bail!(InvalidOperation: "reverse does not accept points"),
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            reverse_line_strings(&array).into_list_array().boxed()
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            reverse_multi_points(&array).into_list_array().boxed()
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(
                REVERSE_HOLES = kwargs.reverse_holes;
                reverse_polygons::<REVERSE_HOLES>(&array)
            )
            .into_list_array()
            .boxed()
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            with_const_flags!(NESTED = kwargs.nested; reverse_multi_line_strings::<NESTED>(&array))
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(
                REVERSE_HOLES = kwargs.reverse_holes, NESTED = kwargs.nested;
                reverse_multi_polygons::<REVERSE_HOLES, NESTED>(&array)
            )
            .into_list_array()
            .boxed()
        }
    })
}

/// See `reverse`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn reverse(inputs: &[Series], kwargs: ReverseKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_reversed(chunk.as_ref(), kind, &kwargs))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
