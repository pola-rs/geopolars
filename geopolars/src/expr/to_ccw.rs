//! Making a geometry run counter-clockwise, with a polygon's holes clockwise.
//! See `to_ccw` in Python for the rules.

use geopolars_compute::native::{
    line_strings_to_ccw, multi_line_strings_to_ccw, multi_polygons_to_ccw, polygons_to_ccw,
};
use geopolars_compute::with_const_flags;
use geopolars_core::{
    downcast, Kind, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
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
pub struct ToCcwKwargs {
    holes_cw: bool,
}

/// The kernel of `kind` on one chunk of the storage.
fn chunk_to_ccw(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &ToCcwKwargs,
) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        Kind::Point | Kind::MultiPoint => {
            polars_bail!(InvalidOperation: "to_ccw does not accept a `{}` column", kind.name())
        }
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            line_strings_to_ccw(&array).into_list_array().boxed()
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(HOLES_CW = kwargs.holes_cw; polygons_to_ccw::<HOLES_CW>(&array))
                .into_list_array()
                .boxed()
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            multi_line_strings_to_ccw(&array).into_list_array().boxed()
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(HOLES_CW = kwargs.holes_cw; multi_polygons_to_ccw::<HOLES_CW>(&array))
                .into_list_array()
                .boxed()
        }
    })
}

/// See `to_ccw`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn to_ccw(inputs: &[Series], kwargs: ToCcwKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_to_ccw(chunk.as_ref(), kind, &kwargs))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
