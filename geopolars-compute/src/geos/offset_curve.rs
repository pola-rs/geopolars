//! Offset curves, forwarded to GEOS.
//!
//! GEOS builds an offset curve from a buffer in 2D, so every output is XY.
//!
//! # Const generic flags
//!
//! ## `EARLY_FINITE_CHECK`
//!
//! The public functions take `early_finite_check`.
//! See [super] for `EARLY_FINITE_CHECK`.

use geopolars_core::{LineStringArray, MultiLineStringArray};
use geos::{Geom, GeometryTypes, JoinStyle};

use super::buffer::{by_distance_via_geos, finite_distance, BufferDistance};
use super::{unless_split, GeosGeometries, ToGeosRows};
use crate::GeoComputeResult;

/// The parameters GEOS's offset curve takes, besides the distance.
pub struct OffsetCurveStyle {
    /// The number of segments per quarter circle.
    /// GEOS raises it to at least 8.
    pub quadrant_segments: i32,
    pub join_style: JoinStyle,
    /// Only used with [`JoinStyle::Mitre`].
    pub mitre_limit: f64,
}

/// The offset curve of every row at `distance`, forwarded to GEOS, as a linestring.
/// A positive distance offsets to the left, relative to the direction of the line.
/// A negative one offsets to the right.
/// The curve keeps the direction of the line.
/// A NaN or infinite distance fails like a GEOS error.
/// A line without length gives an empty linestring.
/// A row that GEOS splits into several linestrings fails with [`crate::GeoComputeError::Split`],
/// like a GEOS error.
pub fn offset_curves_into_line_strings(
    array: &LineStringArray,
    distance: BufferDistance,
    style: &OffsetCurveStyle,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<LineStringArray> {
    offset_curves_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        distance,
        style,
        ignore_per_row_errors,
        false,
    )?
    .into_line_string_array::<false>(ignore_per_row_errors)
}

/// Like [`offset_curves_into_line_strings`], as a multilinestring,
/// for a linestring or multilinestring.
/// A row that GEOS splits into several linestrings gets them all as parts.
pub fn offset_curves_into_multi_line_strings(
    array: &impl ToGeosRows,
    distance: BufferDistance,
    style: &OffsetCurveStyle,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<MultiLineStringArray> {
    offset_curves_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        distance,
        style,
        ignore_per_row_errors,
        true,
    )?
    .into_multi_line_string_array::<false>(ignore_per_row_errors)
}

/// See [`by_distance_via_geos`].
fn offset_curves_via_geos(
    geometries: GeosGeometries,
    distance: BufferDistance,
    style: &OffsetCurveStyle,
    ignore_per_row_errors: bool,
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
        ignore_per_row_errors,
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
