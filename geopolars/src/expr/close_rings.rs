//! Closing the open rings (and linestrings) of a geometry.
//! See `close_rings` in Python for the rules.

use geopolars_core::{CloseRings, SegmentKind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, geometry_series};

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

/// For [`close_rings`]: the input's type.
fn kernel_output_type_close_rings(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    SegmentKind::new("close_rings", describe(field.dtype())?.kind)?;
    Ok(field.clone())
}

/// See `close_rings`.
#[polars_expr(output_type_func=kernel_output_type_close_rings)]
fn close_rings(inputs: &[Series], kwargs: CloseRingsKwargs) -> PolarsResult<Series> {
    let kind = SegmentKind::new("close_rings", describe(inputs[0].dtype())?.kind)?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| geopolars_core::close_rings(chunk.as_ref(), kind, kwargs.into()))
        .collect::<PolarsResult<_>>()?;
    let geo = describe(inputs[0].dtype())?.geo();
    geometry_series(inputs[0].name().clone(), geo, chunks)
}
