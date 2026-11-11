//! Closing the open rings (and linestrings) of a geometry.
//! See `close_rings` in Python for the rules.

use geopolars_compute::native::close_open_rings;
use geopolars_compute::with_const_flags;
use geopolars_core::{
    downcast, Kind, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::construct::closed_rings;
use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::describe;

/// See python
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloseRingsKwargs {
    pub(super) snap_z: bool,
    pub(super) snap_m: bool,
}

/// The kernel of `kind` on one chunk of the storage.
/// `None` if no ring changed.
fn chunk_with_closed_rings(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &CloseRingsKwargs,
) -> PolarsResult<Option<Box<dyn Array>>> {
    let CloseRingsKwargs { snap_z, snap_m } = *kwargs;
    Ok(match kind {
        Kind::Point | Kind::MultiPoint => {
            polars_bail!(InvalidOperation: "close_rings does not accept a `{}` column", kind.name())
        }
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            let z = array.coords.dim().has_z();
            let rows = array.rows.ignoring_validity();
            with_const_flags!(
                Z = z;
                close_open_rings::<Z>(rows, &array.coords, snap_z, snap_m)
            )
            .map(|closed| {
                LineStringArray::from_parts(
                    closed.rows.with_validity(array.rows.validity().cloned()),
                    closed.coords,
                )
                .into_list_array()
                .boxed()
            })
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            closed_rings(&array.rings, &array.coords, *kwargs).map(|(rings, coords)| {
                PolygonArray::from_parts(array.rows, rings, coords)
                    .into_list_array()
                    .boxed()
            })
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            closed_rings(&array.lines, &array.coords, *kwargs).map(|(lines, coords)| {
                MultiLineStringArray::from_parts(array.rows, lines, coords)
                    .into_list_array()
                    .boxed()
            })
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            closed_rings(&array.rings, &array.coords, *kwargs).map(|(rings, coords)| {
                MultiPolygonArray::from_parts(array.rows, array.polygons, rings, coords)
                    .into_list_array()
                    .boxed()
            })
        }
    })
}

/// See `close_rings`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn close_rings(inputs: &[Series], kwargs: CloseRingsKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            Ok(chunk_with_closed_rings(chunk.as_ref(), kind, &kwargs)?
                .unwrap_or_else(|| chunk.clone()))
        })
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
