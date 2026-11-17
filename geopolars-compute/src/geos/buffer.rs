//! Buffers, forwarded to GEOS.
//!
//! GEOS builds a buffer in 2D, so every output is XY.
//!
//! # Const generic flags
//!
//! ## `EARLY_FINITE_CHECK`
//!
//! The public functions take `early_coord_check`.
//! See [super] for `EARLY_FINITE_CHECK`.

use geopolars_views::{GeometryArray, MultiPolygonArray, PolygonArray};
use geos::{BufferParams, Geom, Geometry, GeometryTypes};
pub use geos::{CapStyle, JoinStyle};
use polars_arrow::array::PrimitiveArray;

use super::{unless_split, validity_with_distances, GeosGeometries, ToGeosRows};
use crate::{
    EarlyCoordCheck, GeoComputeError, GeoComputeResult, NewHoles, NewRingOrientation, OnRowError,
};

/// How far away is the buffer?
#[derive(Clone, Copy)]
pub enum BufferDistance<'a> {
    /// One distance for every row.
    Constant(f64),
    /// One distance per row.
    /// A null distance gives a null row.
    PerRow(&'a PrimitiveArray<f64>),
}

/// Which side of a line is buffered
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Both,
    /// The left side, relative to the direction of the line.
    /// The distance has to be at least 0.
    Left,
    /// Like [`Side::Left`], for the right side.
    Right,
}

/// GEOS's `BufferParams`
pub struct BufferStyle {
    /// The number of segments per quarter circle.
    pub quadrant_segments: i32,
    pub cap_style: CapStyle,
    pub join_style: JoinStyle,
    /// Only used with [`JoinStyle::Mitre`].
    pub mitre_limit: f64,
    /// With one side, GEOS ignores `cap_style`.
    pub side: Side,
}

/// Buffers every point, linestring, or polygon into a polygon through GEOS.
/// Implements the Python `buffer` where it gives a polygon, which documents its behaviour.
/// A row that GEOS splits into several polygons therefore fails with [`GeoComputeError::Split`].
/// [`buffer_into_multi_polygons`] keeps them instead.
pub fn buffer_into_polygons(
    array: GeometryArray,
    distance: BufferDistance,
    style: &BufferStyle,
    new_ring_orientation: NewRingOrientation,
    new_holes: NewHoles,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<PolygonArray> {
    let validity = validity_with_distances(array.validity(), distance);
    buffer_via_geos(
        array
            .with_validity(validity)
            .to_geos_rows(on_row_error, early_coord_check)?,
        distance,
        style,
        on_row_error,
        false,
    )?
    .into_polygon_array(false, on_row_error, new_ring_orientation, new_holes)
}

/// Like [`buffer_into_polygons`], but into a multipolygon, for any kind.
/// Implements the Python `buffer` where it gives a multipolygon.
/// A row that GEOS splits into several polygons gets them all as parts.
pub fn buffer_into_multi_polygons(
    array: GeometryArray,
    distance: BufferDistance,
    style: &BufferStyle,
    new_ring_orientation: NewRingOrientation,
    new_holes: NewHoles,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<MultiPolygonArray> {
    let validity = validity_with_distances(array.validity(), distance);
    buffer_via_geos(
        array
            .with_validity(validity)
            .to_geos_rows(on_row_error, early_coord_check)?,
        distance,
        style,
        on_row_error,
        true,
    )?
    .into_multi_polygon_array(false, on_row_error, new_ring_orientation, new_holes)
}

fn buffer_params(style: &BufferStyle) -> GeoComputeResult<BufferParams> {
    Ok(BufferParams::builder()
        .quadrant_segments(style.quadrant_segments)
        .end_cap_style(style.cap_style)
        .join_style(style.join_style)
        .mitre_limit(style.mitre_limit)
        .single_sided(style.side != Side::Both)
        .build()?)
}

/// Every geometry buffered by `distance`, forwarded to GEOS.
/// See [`by_distance_via_geos`].
fn buffer_via_geos(
    geometries: GeosGeometries,
    distance: BufferDistance,
    style: &BufferStyle,
    on_row_error: OnRowError,
    allow_splitting: bool,
) -> GeoComputeResult<GeosGeometries> {
    let params = buffer_params(style)?;
    by_distance_via_geos(
        geometries,
        distance,
        on_row_error,
        |distance| geos_distance(distance, style.side),
        |input, distance| {
            let buffer = input.buffer_with_params(distance, &params)?;
            if allow_splitting {
                return Ok(buffer);
            }
            unless_split(buffer, GeometryTypes::MultiPolygon)
        },
    )
}

/// Every geometry passed to `op` with its distance, forwarded to GEOS.
/// A row with a null distance is already null in `geometries` ([validity_with_distances]).
/// `geos_distance` checks a distance, and gives it as `op` takes it.
/// If `on_row_error=`[OnRowError::Null], a geometry that fails becomes null;
/// otherwise its error is returned.
/// A constant distance that fails would fail every row, so its error is always returned.
/// The size hint stays the input's: it is only a starting capacity.
/// Errors are the exception, so `on_row_error` is checked per row.
pub(super) fn by_distance_via_geos(
    mut geometries: GeosGeometries,
    distance: BufferDistance,
    on_row_error: OnRowError,
    geos_distance: impl Fn(f64) -> GeoComputeResult<f64>,
    op: impl Fn(&Geometry, f64) -> GeoComputeResult<Geometry>,
) -> GeoComputeResult<GeosGeometries> {
    let rows = &mut geometries.geometries;
    match distance {
        BufferDistance::Constant(distance) => {
            let distance = geos_distance(distance)?;
            rows.replace_geometries(on_row_error, |_, input| op(input, distance))?;
        }
        BufferDistance::PerRow(distances) => {
            debug_assert_eq!(distances.len(), rows.len());
            rows.replace_geometries(on_row_error, |i, input| {
                op(input, geos_distance(distances.value(i))?)
            })?;
        }
    }
    Ok(geometries)
}

/// `distance`, or an error if it is not finite.
#[inline(always)]
pub(super) fn finite_distance(distance: f64) -> GeoComputeResult<f64> {
    if !distance.is_finite() {
        return Err(GeoComputeError::NotFiniteDistance { distance });
    }
    Ok(distance)
}

/// `distance` as GEOS takes it, which gives the right side by a negative sign.
/// An error if it is not finite, or negative with a single `side`.
#[inline(always)]
fn geos_distance(distance: f64, side: Side) -> GeoComputeResult<f64> {
    let distance = finite_distance(distance)?;
    match side {
        Side::Both => Ok(distance),
        _ if distance < 0.0 => Err(GeoComputeError::NegativeOneSidedDistance { distance }),
        Side::Left => Ok(distance),
        Side::Right => Ok(-distance),
    }
}
