//! The buffer of any geometry, forwarded to GEOS.

use geopolars_compute::geos::{BufferDistance, BufferStyle, CapStyle, JoinStyle, Side};
use geopolars_core::buffers;
use geopolars_views::{GeoDimension, Kind};
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, Geo};

/// A multipolygon for a multi-kind, or with `allow_splitting`,
/// when `distance` is a column or at most 0.
/// Otherwise a polygon.
/// It is XY, with the input's metadata.
/// Also the kind of the input, and whether the output is a multipolygon.
fn buffered_type(dtype: &DataType, kwargs: &BufferKwargs) -> PolarsResult<(DataType, Kind, bool)> {
    let geo = describe(dtype)?;
    let is_multi = match geo.kind {
        Kind::Point | Kind::LineString | Kind::Polygon => false,
        Kind::MultiPoint | Kind::MultiLineString | Kind::MultiPolygon => true,
    };
    let into_multi =
        is_multi || (kwargs.allow_splitting && kwargs.distance.is_none_or(|d| d <= 0.0));
    let kind = if into_multi {
        Kind::MultiPolygon
    } else {
        Kind::Polygon
    };
    let dtype = Geo::new(kind, GeoDimension::XY, geo.metadata.clone()).dtype();
    Ok((dtype, geo.kind, into_multi))
}

/// For [`buffer`].
fn kernel_output_type_buffer(input_fields: &[Field], kwargs: BufferKwargs) -> PolarsResult<Field> {
    let (dtype, _, _) = buffered_type(input_fields[0].dtype(), &kwargs)?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

#[derive(Deserialize)]
#[serde(remote = "CapStyle", rename_all = "lowercase")]
enum CapStyleDef {
    Round,
    Flat,
    Square,
}

#[derive(Deserialize)]
#[serde(remote = "JoinStyle", rename_all = "lowercase")]
pub(super) enum JoinStyleDef {
    Round,
    Mitre,
    Bevel,
}

#[derive(Deserialize)]
#[serde(remote = "Side", rename_all = "lowercase")]
enum SideDef {
    Both,
    Left,
    Right,
}

/// See python docs for details.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferKwargs {
    /// `None` when the distances are a column, the second input.
    distance: Option<f64>,
    quadrant_segments: i32,
    #[serde(with = "CapStyleDef")]
    cap_style: CapStyle,
    #[serde(with = "JoinStyleDef")]
    join_style: JoinStyle,
    mitre_limit: f64,
    #[serde(with = "SideDef")]
    side: Side,
    allow_splitting: bool,
    ccw_outer: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// See `buffer`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_buffer)]
fn buffer(inputs: &[Series], kwargs: BufferKwargs) -> PolarsResult<Series> {
    let (dtype, kind, into_multi) = buffered_type(inputs[0].dtype(), &kwargs)?;
    let BufferKwargs {
        distance,
        quadrant_segments,
        cap_style,
        join_style,
        mitre_limit,
        side,
        allow_splitting: _,
        ccw_outer,
        ignore_per_row_errors,
        early_finite_check,
    } = kwargs;
    let style = &BufferStyle {
        quadrant_segments,
        cap_style,
        join_style,
        mitre_limit,
        side,
    };
    let chunks = chunks_by_distance(inputs, distance, |rows, distance| {
        buffers(
            rows,
            kind,
            into_multi,
            distance,
            style,
            ccw_outer,
            ignore_per_row_errors,
            early_finite_check,
        )
    })?;
    // SAFETY: the buffers are XY polygons or multipolygons, as `dtype` says.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}

/// Every chunk of the geometries, the first input, through `chunk`.
/// `distance` is `None` when the distances are a column, the second input.
/// Then the geometries are rechunked into one chunk, to line up with the distances.
pub(super) fn chunks_by_distance<A: Array>(
    inputs: &[Series],
    distance: Option<f64>,
    chunk: impl Fn(&dyn Array, BufferDistance) -> PolarsResult<A>,
) -> PolarsResult<Vec<ArrayRef>> {
    let storage = inputs[0].ext()?.storage();
    match distance {
        Some(distance) => storage
            .chunks()
            .iter()
            .map(|rows| {
                Ok(Box::new(chunk(rows.as_ref(), BufferDistance::Constant(distance))?) as ArrayRef)
            })
            .collect(),
        None => {
            let storage = storage.rechunk();
            // Python broadcast the distances to one per geometry.
            let distances = inputs[1].rechunk();
            polars_ensure!(
                distances.len() == storage.len(),
                ShapeMismatch: "got {} geometries but {} distances",
                storage.len(),
                distances.len()
            );
            let distance = BufferDistance::PerRow(distances.f64()?.downcast_as_array());
            let rows = storage.chunks()[0].as_ref();
            Ok(vec![Box::new(chunk(rows, distance)?)])
        }
    }
}
