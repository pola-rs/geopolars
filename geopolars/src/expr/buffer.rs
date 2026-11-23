//! The buffer of any geometry, forwarded to GEOS.
//! Implements the Python `buffer`.

use crate::common_arguments::{
    EarlyCoordCheckDef, NewHolesDef, NewRingOrientationDef, OnRowErrorDef, PromoteToMulti,
};
use geopolars_compute::geos::{BufferDistance, BufferStyle, CapStyle, JoinStyle, Side};
use geopolars_compute::{EarlyCoordCheck, NewHoles, NewRingOrientation, OnRowError};
use geopolars_core::buffers;
use geopolars_views::{GeoDimension, GeometryArray, Kind};
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, geometry_series, Geo};

/// Used for Python `buffer`.
/// The type of the buffers of a `dtype` column
/// Also the kind of the input, and whether the output is a multipolygon.
fn buffered_type(dtype: &DataType, kwargs: &BufferKwargs) -> PolarsResult<(Geo, Kind, bool)> {
    let geo = describe(dtype)?;
    // Only a distance of at most 0 can split, and a column's sign is only known per row.
    let can_split = kwargs.distance.is_none_or(|d| d <= 0.0);
    let into_multi = kwargs
        .promote_to_multi
        .output_is_multi(geo.kind.is_multi(), can_split);
    let kind = if into_multi {
        Kind::MultiPolygon
    } else {
        Kind::Polygon
    };
    let buffered = Geo::new(kind, GeoDimension::XY, geo.metadata.clone());
    Ok((buffered, geo.kind, into_multi))
}

fn kernel_output_type_buffer(input_fields: &[Field], kwargs: BufferKwargs) -> PolarsResult<Field> {
    let (buffered, _, _) = buffered_type(input_fields[0].dtype(), &kwargs)?;
    Ok(Field::new(input_fields[0].name().clone(), buffered.dtype()))
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

/// The keyword arguments of the Python `buffer`.
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
    promote_to_multi: PromoteToMulti,
    #[serde(with = "NewRingOrientationDef")]
    new_ring_orientation: NewRingOrientation,
    #[serde(with = "NewHolesDef")]
    new_holes: NewHoles,
    #[serde(with = "OnRowErrorDef")]
    on_row_error: OnRowError,
    #[serde(with = "EarlyCoordCheckDef")]
    early_coord_check: EarlyCoordCheck,
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_buffer)]
fn buffer(inputs: &[Series], kwargs: BufferKwargs) -> PolarsResult<Series> {
    let (buffered, kind, into_multi) = buffered_type(inputs[0].dtype(), &kwargs)?;
    let BufferKwargs {
        distance,
        quadrant_segments,
        cap_style,
        join_style,
        mitre_limit,
        side,
        promote_to_multi: _,
        new_ring_orientation,
        new_holes,
        on_row_error,
        early_coord_check,
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
            new_ring_orientation,
            new_holes,
            on_row_error,
            early_coord_check,
        )
    })?;
    geometry_series(inputs[0].name().clone(), buffered, chunks)
}

/// Every chunk of the geometries, the first input, through `chunk`.
/// `distance` is `None` when the distances are a column, the second input.
/// Then the geometries are rechunked into one chunk, to line up with the distances.
pub(super) fn chunks_by_distance(
    inputs: &[Series],
    distance: Option<f64>,
    chunk: impl Fn(&dyn Array, BufferDistance) -> PolarsResult<GeometryArray>,
) -> PolarsResult<Vec<GeometryArray>> {
    let storage = inputs[0].ext()?.storage();
    match distance {
        Some(distance) => storage
            .chunks()
            .iter()
            .map(|rows| chunk(rows.as_ref(), BufferDistance::Constant(distance)))
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
            Ok(vec![chunk(rows, distance)?])
        }
    }
}
