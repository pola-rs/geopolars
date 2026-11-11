//! The offset curve of a linestring or multilinestring, forwarded to GEOS.

use geopolars_compute::geos::{
    offset_curves_into_line_strings, offset_curves_into_multi_line_strings, BufferDistance,
    JoinStyle, OffsetCurveStyle,
};
use geopolars_core::{downcast, GeoDimension, Kind, LineStringArray, MultiLineStringArray};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::buffer::{chunks_by_distance, JoinStyleDef};
use super::IntoPolarsResult;
use crate::geoarrow::{describe, Geo};

/// Python refuses any other kind, so it never gets here.
fn not_a_line(dtype: &DataType) -> PolarsError {
    polars_err!(
        SchemaMismatch:
        "offset_curve expects a linestring or multilinestring column, got: {}",
        dtype
    )
}

/// A multilinestring for a multilinestring, or with `allow_splitting`.
/// Otherwise a linestring.
/// It is XY, with the input's metadata.
/// Also the kind of the input, and whether the output is a multilinestring.
fn offset_curve_type(
    dtype: &DataType,
    kwargs: &OffsetCurveKwargs,
) -> PolarsResult<(DataType, Kind, bool)> {
    let geo = describe(dtype)?;
    let into_multi = match geo.kind {
        Kind::LineString => kwargs.allow_splitting,
        Kind::MultiLineString => true,
        Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon => {
            return Err(not_a_line(dtype))
        }
    };
    let kind = if into_multi {
        Kind::MultiLineString
    } else {
        Kind::LineString
    };
    let dtype = Geo::new(kind, GeoDimension::XY, geo.metadata.clone()).dtype();
    Ok((dtype, geo.kind, into_multi))
}

/// For [`offset_curve`].
fn kernel_output_type_offset_curve(
    input_fields: &[Field],
    kwargs: OffsetCurveKwargs,
) -> PolarsResult<Field> {
    let (dtype, _, _) = offset_curve_type(input_fields[0].dtype(), &kwargs)?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
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

/// The offset curves of one chunk of `kind` geometries.
/// If `into_multi=true`, as multilinestrings; otherwise as linestrings.
fn chunk_offset_curves(
    chunk: &dyn Array,
    kind: Kind,
    into_multi: bool,
    distance: BufferDistance,
    kwargs: &OffsetCurveKwargs,
) -> PolarsResult<ListArray<i64>> {
    let OffsetCurveKwargs {
        distance: _,
        quadrant_segments,
        join_style,
        mitre_limit,
        allow_splitting: _,
        ignore_per_row_errors,
        early_finite_check,
    } = *kwargs;
    let style = &OffsetCurveStyle {
        quadrant_segments,
        join_style,
        mitre_limit,
    };
    let (ignore, early) = (ignore_per_row_errors, early_finite_check);
    match (kind, into_multi) {
        (Kind::LineString, false) => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            offset_curves_into_line_strings(&array, distance, style, ignore, early)
                .map(LineStringArray::into_list_array)
        }
        (Kind::LineString, true) => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            offset_curves_into_multi_line_strings(&array, distance, style, ignore, early)
                .map(MultiLineStringArray::into_list_array)
        }
        (Kind::MultiLineString, _) => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            offset_curves_into_multi_line_strings(&array, distance, style, ignore, early)
                .map(MultiLineStringArray::into_list_array)
        }
        (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon, _) => {
            unreachable!("refused by `offset_curve_type`")
        }
    }
    .into_polars()
}

/// See `offset_curve`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_offset_curve)]
fn offset_curve(inputs: &[Series], kwargs: OffsetCurveKwargs) -> PolarsResult<Series> {
    let (dtype, kind, into_multi) = offset_curve_type(inputs[0].dtype(), &kwargs)?;
    let chunks = chunks_by_distance(inputs, kwargs.distance, |rows, distance| {
        chunk_offset_curves(rows, kind, into_multi, distance, &kwargs)
    })?;
    // SAFETY: the offset curves are XY linestrings or multilinestrings, as `dtype` says.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
