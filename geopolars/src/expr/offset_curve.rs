//! The offset curve of a linestring or multilinestring, forwarded to GEOS.

use geopolars_compute::geos::{JoinStyle, OffsetCurveStyle};
use geopolars_core::{offset_curves, LineKind};
use geopolars_views::{GeoDimension, Kind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::buffer::{chunks_by_distance, JoinStyleDef};
use crate::geoarrow::{describe, geometry_series, Geo};

/// The type of the offset curves of a `dtype` column: XY, with the input's metadata.
/// Also gives the input's kind, and whether the curves are multilinestrings
/// (for a multilinestring, or with `allow_splitting=true`).
/// Any other kind is refused here, but Python already refuses it before calling the plugin.
fn offset_curve_type(
    dtype: &DataType,
    kwargs: &OffsetCurveKwargs,
) -> PolarsResult<(Geo, LineKind, bool)> {
    let geo = describe(dtype)?;
    let line_kind = LineKind::new("offset_curve", geo.kind)?;
    let into_multi = match line_kind {
        LineKind::LineString => kwargs.allow_splitting,
        LineKind::MultiLineString => true,
    };
    let kind = if into_multi {
        Kind::MultiLineString
    } else {
        Kind::LineString
    };
    let curve = Geo::new(kind, GeoDimension::XY, geo.metadata.clone());
    Ok((curve, line_kind, into_multi))
}

/// For [`offset_curve`].
fn kernel_output_type_offset_curve(
    input_fields: &[Field],
    kwargs: OffsetCurveKwargs,
) -> PolarsResult<Field> {
    let (curve, _, _) = offset_curve_type(input_fields[0].dtype(), &kwargs)?;
    Ok(Field::new(input_fields[0].name().clone(), curve.dtype()))
}

/// See python docs for details.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OffsetCurveKwargs {
    /// `None` when the distances are a column, the second input.
    distance: Option<f64>,
    quadrant_segments: i32,
    #[serde(with = "JoinStyleDef")]
    join_style: JoinStyle,
    mitre_limit: f64,
    allow_splitting: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// See `offset_curve`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_offset_curve)]
fn offset_curve(inputs: &[Series], kwargs: OffsetCurveKwargs) -> PolarsResult<Series> {
    let (curve, kind, into_multi) = offset_curve_type(inputs[0].dtype(), &kwargs)?;
    let OffsetCurveKwargs {
        distance,
        quadrant_segments,
        join_style,
        mitre_limit,
        allow_splitting: _,
        ignore_per_row_errors,
        early_finite_check,
    } = kwargs;
    let style = &OffsetCurveStyle {
        quadrant_segments,
        join_style,
        mitre_limit,
    };
    let chunks = chunks_by_distance(inputs, distance, |rows, distance| {
        offset_curves(
            rows,
            kind,
            into_multi,
            distance,
            style,
            ignore_per_row_errors,
            early_finite_check,
        )
    })?;
    geometry_series(inputs[0].name().clone(), curve, chunks)
}
