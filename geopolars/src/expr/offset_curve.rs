//! The offset curve of a linestring or multilinestring, forwarded to GEOS.
//! Implements the Python `offset_curve`.

use crate::common_arguments::{EarlyCoordCheckDef, OnRowErrorDef};
use geopolars_compute::geos::{JoinStyle, OffsetCurveStyle};
use geopolars_compute::{EarlyCoordCheck, OnRowError};
use geopolars_core::{offset_curves, LineKind};
use geopolars_views::{GeoDimension, Kind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::buffer::{chunks_by_distance, JoinStyleDef};
use crate::geoarrow::{describe, geometry_series, Geo};

/// The type of the offset curves of a `dtype` column, as the Python `offset_curve` gives it.
/// Also gives the input's kind, and whether the curves are multilinestrings.
fn offset_curve_type(
    dtype: &DataType,
    kwargs: &OffsetCurveKwargs,
) -> PolarsResult<(Geo, LineKind, bool)> {
    // non-string kinds should already be refused in python,
    // but second check never hurt nobody.
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

fn kernel_output_type_offset_curve(
    input_fields: &[Field],
    kwargs: OffsetCurveKwargs,
) -> PolarsResult<Field> {
    let (curve, _, _) = offset_curve_type(input_fields[0].dtype(), &kwargs)?;
    Ok(Field::new(input_fields[0].name().clone(), curve.dtype()))
}

/// The keyword arguments of the Python `offset_curve`.
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
    #[serde(with = "OnRowErrorDef")]
    on_row_error: OnRowError,
    #[serde(with = "EarlyCoordCheckDef")]
    early_coord_check: EarlyCoordCheck,
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_offset_curve)]
fn offset_curve(inputs: &[Series], kwargs: OffsetCurveKwargs) -> PolarsResult<Series> {
    let (curve, kind, into_multi) = offset_curve_type(inputs[0].dtype(), &kwargs)?;
    let OffsetCurveKwargs {
        distance,
        quadrant_segments,
        join_style,
        mitre_limit,
        allow_splitting: _,
        on_row_error,
        early_coord_check,
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
            on_row_error,
            early_coord_check,
        )
    })?;
    geometry_series(inputs[0].name().clone(), curve, chunks)
}
