//! Closing the open rings (and linestrings) of a geometry.
//! See `close_rings` in Python for the rules.

use geopolars_core::CloseRings;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::describe;

/// See python
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloseRingsKwargs {
    snap_z: bool,
    snap_m: bool,
}

impl From<CloseRingsKwargs> for CloseRings {
    fn from(kwargs: CloseRingsKwargs) -> Self {
        let CloseRingsKwargs { snap_z, snap_m } = kwargs;
        Self { snap_z, snap_m }
    }
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
            Ok(
                geopolars_core::close_rings(chunk.as_ref(), kind, kwargs.into())?
                    .unwrap_or_else(|| chunk.clone()),
            )
        })
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
