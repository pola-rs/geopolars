//! The point at a distance along a line, forwarded to GEOS.
//!
//! GEOS interpolates z and m too, so every output keeps the input's dimension.
//!
//! # Const generic flags
//!
//! ## `EARLY_FINITE_CHECK`
//!
//! The public functions take `early_coord_check`.
//! See [super] for `EARLY_FINITE_CHECK`.

use geopolars_views::{LineStringArray, MultiLineStringArray, PointArray};
use geos::Geom;

use super::buffer::{by_distance_via_geos, finite_distance, BufferDistance};
use super::linestrings::multi_line_strings_to_geos_rows;
use super::{validity_with_distances, GeosGeometries, ToGeosRows};
use crate::{EarlyCoordCheck, GeoComputeResult, OnRowError};

/// What the distance along the line is measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolateMode {
    /// The units of the coordinates.
    Absolute,
    /// A fraction of the length of the line.
    Normalized,
}

/// Gets the point at `distance` along every linestring through GEOS.
/// Implements the Python `line_interpolate_point` for linestrings, which documents its behaviour.
pub fn interpolate_line_strings(
    mut array: LineStringArray,
    distance: BufferDistance,
    mode: InterpolateMode,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<PointArray> {
    let validity = validity_with_distances(array.rows.validity(), distance);
    array.rows = array.rows.with_validity(validity);
    interpolate_via_geos(
        array.to_geos_rows(on_row_error, early_coord_check)?,
        distance,
        mode,
        on_row_error,
    )?
    .into_point_array(array.coords.dim())
}

/// Like [`interpolate_line_strings`], for multilinestrings, walking the parts one after the other.
pub fn interpolate_multi_line_strings(
    mut array: MultiLineStringArray,
    distance: BufferDistance,
    mode: InterpolateMode,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<PointArray> {
    let validity = validity_with_distances(array.rows.validity(), distance);
    array.rows = array.rows.with_validity(validity);
    // An empty part is skipped,
    // as GEOS reads a vertex of every part it walks, beyond the end of an empty one.
    interpolate_via_geos(
        multi_line_strings_to_geos_rows(&array, true, None, on_row_error, early_coord_check)?,
        distance,
        mode,
        on_row_error,
    )?
    .into_point_array(array.coords.dim())
}

/// See [`by_distance_via_geos`].
fn interpolate_via_geos(
    geometries: GeosGeometries,
    distance: BufferDistance,
    mode: InterpolateMode,
    on_row_error: OnRowError,
) -> GeoComputeResult<GeosGeometries> {
    match mode {
        InterpolateMode::Absolute => by_distance_via_geos(
            geometries,
            distance,
            on_row_error,
            finite_distance,
            |input, distance| Ok(input.interpolate(distance)?),
        ),
        InterpolateMode::Normalized => by_distance_via_geos(
            geometries,
            distance,
            on_row_error,
            finite_distance,
            |input, fraction| Ok(input.interpolate_normalized(fraction)?),
        ),
    }
}
