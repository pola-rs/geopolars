//! The simplification of a linestring or polygon, or their multi-variants, forwarded to GEOS.

use geopolars_core::SegmentKind;
use geopolars_views::GeoDimension;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, geometry_series, Geo};

/// The type of the simplified geometries of a `dtype` column.
/// This is drops the m on the inputs kind.
fn simplified_type(dtype: &DataType) -> PolarsResult<(Geo, SegmentKind)> {
    let geo = describe(dtype)?;
    let kind = SegmentKind::new("simplify", geo.kind)?;
    let dim = if geo.dim.has_z() {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let simplified = Geo::new(kind.kind(), dim, geo.metadata.clone());
    Ok((simplified, kind))
}

/// For [`simplify`].
fn kernel_output_type_simplify(input_fields: &[Field]) -> PolarsResult<Field> {
    let (simplified, _) = simplified_type(input_fields[0].dtype())?;
    Ok(Field::new(
        input_fields[0].name().clone(),
        simplified.dtype(),
    ))
}

/// See python docs for details.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimplifyKwargs {
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// See `simplify`.
#[polars_expr(output_type_func=kernel_output_type_simplify)]
fn simplify(inputs: &[Series], kwargs: SimplifyKwargs) -> PolarsResult<Series> {
    let (simplified, kind) = simplified_type(inputs[0].dtype())?;
    let SimplifyKwargs {
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
        early_finite_check,
    } = kwargs;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            geopolars_core::simplify(
                chunk.as_ref(),
                kind,
                tolerance,
                preserve_topology,
                ignore_per_row_errors,
                early_finite_check,
            )
        })
        .collect::<PolarsResult<_>>()?;
    geometry_series(inputs[0].name().clone(), simplified, chunks)
}
