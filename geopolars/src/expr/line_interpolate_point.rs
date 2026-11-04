use geopolars_compute::geos::{
    interpolate_line_strings, interpolate_multi_line_strings, BufferDistance, InterpolateMode,
};
use geopolars_core::{downcast, LineStringArray, MultiLineStringArray, PointArray};
use polars::prelude::*;
use polars_arrow::array::{Array, StructArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::buffer::chunks_by_distance;
use super::IntoPolarsResult;
use crate::geoarrow::{describe, Geo, Kind};

/// A point with the input's dimension and metadata.
/// Also the kind of the input.
fn line_interpolate_point_type(dtype: &DataType) -> PolarsResult<(DataType, Kind)> {
    let geo = describe(dtype)?;
    match geo.kind {
        Kind::LineString | Kind::MultiLineString => {}
        // Python refuses any other kind, so it never gets here.
        Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon => polars_bail!(
            SchemaMismatch:
            "line_interpolate_point expects a linestring or multilinestring column, got: {}",
            dtype
        ),
    }
    let dtype = Geo::new(Kind::Point, geo.dim, geo.metadata.clone()).dtype();
    Ok((dtype, geo.kind))
}

/// For [`line_interpolate_point`].
fn kernel_output_type_line_interpolate_point(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = line_interpolate_point_type(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
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

/// The interpolated points of one chunk of `kind` geometries.
fn chunk_interpolated(
    chunk: &dyn Array,
    kind: Kind,
    distance: BufferDistance,
    kwargs: &LineInterpolatePointKwargs,
) -> PolarsResult<StructArray> {
    let LineInterpolatePointKwargs {
        distance: _,
        mode,
        ignore_per_row_errors,
        early_finite_check,
    } = *kwargs;
    let (ignore, early) = (ignore_per_row_errors, early_finite_check);
    match kind {
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            interpolate_line_strings(&array, distance, mode, ignore, early)
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            interpolate_multi_line_strings(&array, distance, mode, ignore, early)
        }
        Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon => {
            unreachable!("refused by `line_interpolate_point_type`")
        }
    }
    .map(PointArray::into_struct_array)
    .into_polars()
}

/// See `line_interpolate_point`.
#[polars_expr(output_type_func=kernel_output_type_line_interpolate_point)]
fn line_interpolate_point(
    inputs: &[Series],
    kwargs: LineInterpolatePointKwargs,
) -> PolarsResult<Series> {
    let (dtype, kind) = line_interpolate_point_type(inputs[0].dtype())?;
    let chunks = chunks_by_distance(inputs, kwargs.distance, |rows, distance| {
        chunk_interpolated(rows, kind, distance, &kwargs)
    })?;
    // SAFETY: the points have the input's dimension, as `dtype` says.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
