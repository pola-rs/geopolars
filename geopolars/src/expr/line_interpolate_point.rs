use geopolars_compute::geos::InterpolateMode;
use geopolars_core::{interpolated_points, LineKind};
use geopolars_views::Kind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::buffer::chunks_by_distance;
use crate::geoarrow::{describe, geometry_series, Geo};

/// A point with the input's dimension and metadata.
/// Also the kind of the input.
fn line_interpolate_point_type(dtype: &DataType) -> PolarsResult<(Geo, LineKind)> {
    let geo = describe(dtype)?;
    // Python already refuses any other kind before it reaches the plugin.
    let kind = LineKind::new("line_interpolate_point", geo.kind)?;
    let point = Geo::new(Kind::Point, geo.dim, geo.metadata.clone());
    Ok((point, kind))
}

/// For [`line_interpolate_point`].
fn kernel_output_type_line_interpolate_point(input_fields: &[Field]) -> PolarsResult<Field> {
    let (point, _) = line_interpolate_point_type(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), point.dtype()))
}

#[derive(Deserialize)]
#[serde(remote = "InterpolateMode", rename_all = "lowercase")]
enum InterpolateModeDef {
    Absolute,
    Normalized,
}

/// See python docs for details.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineInterpolatePointKwargs {
    /// `None` when the distances are a column, the second input.
    distance: Option<f64>,
    #[serde(with = "InterpolateModeDef")]
    mode: InterpolateMode,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// See `line_interpolate_point`.
#[polars_expr(output_type_func=kernel_output_type_line_interpolate_point)]
fn line_interpolate_point(
    inputs: &[Series],
    kwargs: LineInterpolatePointKwargs,
) -> PolarsResult<Series> {
    let (point, kind) = line_interpolate_point_type(inputs[0].dtype())?;
    let LineInterpolatePointKwargs {
        distance,
        mode,
        ignore_per_row_errors,
        early_finite_check,
    } = kwargs;
    let chunks = chunks_by_distance(inputs, distance, |rows, distance| {
        interpolated_points(
            rows,
            kind,
            distance,
            mode,
            ignore_per_row_errors,
            early_finite_check,
        )
    })?;
    geometry_series(inputs[0].name().clone(), point, chunks)
}
