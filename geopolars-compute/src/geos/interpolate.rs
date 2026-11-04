//! The point at a distance along a line, forwarded to GEOS.
//!
//! GEOS interpolates z and m too, so every output keeps the input's dimension.
//!
//! # Const generic flags
//!
//! ## `EARLY_FINITE_CHECK`
//!
//! The public functions take `early_finite_check`.
//! See [super] for `EARLY_FINITE_CHECK`.

use geopolars_core::{LineStringArray, MultiLineStringArray, PointArray};
use geos::Geom;

use super::buffer::{by_distance_via_geos, finite_distance, BufferDistance};
use super::linestrings::multi_line_strings_to_geos_rows;
use super::{GeosGeometries, ToGeosRows};
use crate::GeoComputeResult;

/// What the distance along the line is measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolateMode {
    /// The units of the coordinates.
    Absolute,
    /// A fraction of the length of the line.
    Normalized,
}

/// The point of every row at `distance` along it, forwarded to GEOS.
/// A negative distance counts back from the end.
/// One beyond either end gives that end.
/// A NaN or infinite distance fails like a GEOS error.
/// A line without vertices gives an empty point.
pub fn interpolate_line_strings(
    array: &LineStringArray,
    distance: BufferDistance,
    mode: InterpolateMode,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<PointArray> {
    interpolate_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        distance,
        mode,
        ignore_per_row_errors,
    )?
    .into_point_array(array.coords.dim(), ignore_per_row_errors)
}

/// Like [`interpolate_line_strings`], along the parts one after the other.
/// The gaps between the parts don't count.
/// An empty part is skipped:
/// GEOS reads a vertex of every part it walks, beyond the end of an empty one.
pub fn interpolate_multi_line_strings(
    array: &MultiLineStringArray,
    distance: BufferDistance,
    mode: InterpolateMode,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<PointArray> {
    interpolate_via_geos(
        multi_line_strings_to_geos_rows(array, true, ignore_per_row_errors, early_finite_check)?,
        distance,
        mode,
        ignore_per_row_errors,
    )?
    .into_point_array(array.coords.dim(), ignore_per_row_errors)
}

/// See [`by_distance_via_geos`].
fn interpolate_via_geos(
    geometries: GeosGeometries,
    distance: BufferDistance,
    mode: InterpolateMode,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<GeosGeometries> {
    match mode {
        InterpolateMode::Absolute => by_distance_via_geos(
            geometries,
            distance,
            ignore_per_row_errors,
            finite_distance,
            |input, distance| input.interpolate(distance),
        ),
        InterpolateMode::Normalized => by_distance_via_geos(
            geometries,
            distance,
            ignore_per_row_errors,
            finite_distance,
            |input, fraction| input.interpolate_normalized(fraction),
        ),
    }
}
