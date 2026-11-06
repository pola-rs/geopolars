//! Buffers, forwarded to GEOS.
//!
//! GEOS builds a buffer in 2D, so every output is XY.
//!
//! # Const generic flags
//!
//! ## `EARLY_FINITE_CHECK`
//!
//! The public functions take `early_finite_check`.
//! See [super] for `EARLY_FINITE_CHECK`.

use geopolars_core::{GeometryArray, MultiPolygonArray, PolygonArray};
use geos::{BufferParams, Geom, Geometry};
pub use geos::{CapStyle, JoinStyle};
use polars_arrow::array::PrimitiveArray;

use super::{GeosGeometries, GeosRow, ToGeosRows};
use crate::{GeoComputeError, GeoComputeResult};

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

/// Every row buffered by `distance`, forwarded to GEOS, as a polygon.
/// A NaN or infinite distance fails like a GEOS error,
/// as does a negative one with a single `side`.
/// This is for a point, linestring or polygon.
/// A distance of at most 0 gives an empty polygon for a point or linestring.
/// A negative distance shrinks a polygon.
/// A row that GEOS buffers into several polygons fails like a GEOS error.
/// For a single kind, that only happens with a negative distance.
/// If `ignore_per_row_errors=true`, a row GEOS fails on is null;
/// otherwise its error is returned.
/// If `ccw_outer=true`, every ring is reversed, so the exterior runs counter-clockwise.
/// Otherwise it runs clockwise, as GEOS gives it.
pub fn buffer_into_polygons(
    array: &GeometryArray,
    distance: BufferDistance,
    style: &BufferStyle,
    ccw_outer: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<PolygonArray> {
    buffer_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        distance,
        style,
        ignore_per_row_errors,
    )?
    .into_polygon_array(false, ignore_per_row_errors, ccw_outer)
}

/// Like [`buffer_into_polygons`], as a multipolygon, for any kind.
/// Parts whose buffers overlap are merged into one.
/// A row that GEOS buffers into several polygons gets them all as parts.
pub fn buffer_into_multi_polygons(
    array: &GeometryArray,
    distance: BufferDistance,
    style: &BufferStyle,
    ccw_outer: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<MultiPolygonArray> {
    buffer_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        distance,
        style,
        ignore_per_row_errors,
    )?
    .into_multi_polygon_array(false, ignore_per_row_errors, ccw_outer)
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
    ignore_per_row_errors: bool,
) -> GeoComputeResult<GeosGeometries> {
    let params = buffer_params(style)?;
    by_distance_via_geos(
        geometries,
        distance,
        ignore_per_row_errors,
        |distance| geos_distance(distance, style.side),
        |input, distance| input.buffer_with_params(distance, &params),
    )
}

/// Every geometry passed to `op` with its distance, forwarded to GEOS.
/// `geos_distance` checks a distance, and gives it as `op` takes it.
/// If `ignore_per_row_errors=true`, a geometry that fails becomes null;
/// otherwise its error is returned.
/// A constant distance that fails would fail every row, so its error is always returned.
/// The size hint stays the input's: it is only a starting capacity.
/// Errors are the exception, so `ignore_per_row_errors` is checked per row.
pub(super) fn by_distance_via_geos(
    mut geometries: GeosGeometries,
    distance: BufferDistance,
    ignore_per_row_errors: bool,
    geos_distance: impl Fn(f64) -> GeoComputeResult<f64>,
    op: impl Fn(&Geometry, f64) -> Result<Geometry, geos::Error>,
) -> GeoComputeResult<GeosGeometries> {
    let apply = |input: &Geometry, geos_distance: f64| -> GeoComputeResult<GeosRow> {
        match op(input, geos_distance) {
            Ok(output) => Ok(GeosRow::Geometry(output)),
            Err(e) if !ignore_per_row_errors => Err(e.into()),
            Err(_) => Ok(GeosRow::Null),
        }
    };
    match distance {
        BufferDistance::Constant(distance) => {
            let distance = geos_distance(distance)?;
            for geometry in &mut geometries.geometries {
                if let GeosRow::Geometry(input) = geometry {
                    *geometry = apply(input, distance)?;
                }
            }
        }
        BufferDistance::PerRow(distances) => {
            debug_assert_eq!(distances.len(), geometries.geometries.len());
            for (geometry, distance) in geometries.geometries.iter_mut().zip(distances.iter()) {
                let Some(&distance) = distance else {
                    *geometry = GeosRow::Null;
                    continue;
                };
                let GeosRow::Geometry(input) = geometry else {
                    continue;
                };
                *geometry = match geos_distance(distance) {
                    Ok(distance) => apply(input, distance)?,
                    Err(e) if !ignore_per_row_errors => return Err(e),
                    Err(_) => GeosRow::Null,
                };
            }
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
