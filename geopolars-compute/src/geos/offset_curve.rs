//! Offset curves, forwarded to GEOS.
//!
//! GEOS builds an offset curve from a buffer in 2D, so every output is XY.

use geopolars_views::{GeometryArray, LineStringArray, MultiLineStringArray};
use geos::{Geom, GeometryTypes, JoinStyle};

use super::buffer::{by_distance_via_geos, finite_distance, BufferDistance};
use super::{
    unless_split, validity_with_distances, GeometryOrientation, GeosGeometries, ToGeosRows,
};
use crate::{EarlyCoordCheck, GeoComputeResult, OnRowError};

/// The parameters GEOS's offset curve takes, besides the distance.
pub struct OffsetCurveStyle {
    /// The number of segments per quarter circle.
    /// GEOS raises it to at least 8.
    pub quadrant_segments: i32,
    pub join_style: JoinStyle,
    /// Only used with [`JoinStyle::Mitre`].
    pub mitre_limit: f64,
}

/// Offsets every linestring into a linestring through GEOS.
/// Implements the Python `offset_curve` with `allow_splitting=false`, which documents its behaviour.
/// A row that GEOS splits into several linestrings therefore fails with [`crate::GeoComputeError::Split`].
/// [`offset_curves_into_multi_line_strings`] keeps them instead.
pub fn offset_curves_into_line_strings(
    mut array: LineStringArray,
    distance: BufferDistance,
    style: &OffsetCurveStyle,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<LineStringArray> {
    let validity = validity_with_distances(array.rows.validity(), distance);
    array.rows = array.rows.with_validity(validity);
    offset_curves_via_geos(
        array.to_geos_rows(on_row_error, early_coord_check)?,
        distance,
        style,
        on_row_error,
        false,
    )?
    .into_line_string_array::<false>(on_row_error)
}

/// Like [`offset_curves_into_line_strings`], but into a multilinestring,
/// for a linestring or multilinestring.
/// Implements the Python `offset_curve` where it gives a multilinestring.
/// A row that GEOS splits into several linestrings gets them all as parts.
pub fn offset_curves_into_multi_line_strings(
    array: GeometryArray,
    distance: BufferDistance,
    style: &OffsetCurveStyle,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<MultiLineStringArray> {
    let validity = validity_with_distances(array.validity(), distance);
    offset_curves_via_geos(
        array
            .with_validity(validity)
            .to_geos_rows(on_row_error, early_coord_check)?,
        distance,
        style,
        on_row_error,
        true,
    )?
    .into_multi_line_string_array::<false>(on_row_error)
}

/// See [`by_distance_via_geos`].
fn offset_curves_via_geos(
    geometries: GeosGeometries,
    distance: BufferDistance,
    style: &OffsetCurveStyle,
    on_row_error: OnRowError,
    allow_splitting: bool,
) -> GeoComputeResult<GeosGeometries> {
    let OffsetCurveStyle {
        quadrant_segments,
        join_style,
        mitre_limit,
    } = *style;
    by_distance_via_geos(
        geometries,
        distance,
        // Lines have no rings.
        GeometryOrientation::UNKNOWN,
        on_row_error,
        finite_distance,
        |input, distance| {
            let curve = input.offset_curve(distance, quadrant_segments, join_style, mitre_limit)?;
            if allow_splitting {
                return Ok(curve);
            }
            unless_split(curve, GeometryTypes::MultiLineString)
        },
    )
}
