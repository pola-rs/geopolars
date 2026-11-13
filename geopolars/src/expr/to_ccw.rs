//! Making a geometry run counter-clockwise, with a polygon's holes clockwise.
//! See `to_ccw` in Python for the rules.

use geopolars_core::SegmentKind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, geometry_series};

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToCcwKwargs {
    holes_cw: bool,
}

/// is the input's type.
fn kernel_output_type_to_ccw(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    SegmentKind::new("to_ccw", describe(field.dtype())?.kind)?;
    Ok(field.clone())
}

/// See `to_ccw`.
#[polars_expr(output_type_func=kernel_output_type_to_ccw)]
fn to_ccw(inputs: &[Series], kwargs: ToCcwKwargs) -> PolarsResult<Series> {
    let kind = SegmentKind::new("to_ccw", describe(inputs[0].dtype())?.kind)?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| geopolars_core::to_ccw(chunk.as_ref(), kind, kwargs.holes_cw))
        .collect::<PolarsResult<_>>()?;
    let geo = describe(inputs[0].dtype())?.geo();
    geometry_series(inputs[0].name().clone(), geo, chunks)
}
