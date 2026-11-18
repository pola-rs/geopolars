//! Simplification, forwarded to GEOS.
//! Implements the Python `simplify`, which documents its behaviour.
//! Every output keeps the input's z, but drops m.
//!
//! # Const generic flags
//!
//! ## `PRESERVE_TOPOLOGY`
//!
//! Every function here is generic over `PRESERVE_TOPOLOGY`, or takes it as `preserve_topology`.
//! If `PRESERVE_TOPOLOGY=false`, GEOS uses Douglas-Peucker.
//! It keeps m, but m does not reach the output yet (see the TODO).
//! If `PRESERVE_TOPOLOGY=true`, GEOS's topology-preserving simplifier is used.

use geopolars_views::{LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray};

use super::{GeosGeometries, GeosRows, ToGeosRows};
use crate::{EarlyCoordCheck, GeoComputeResult, NewHoles, NewRingOrientation, OnRowError};

/// Simplifies every linestring to within `tolerance` through GEOS.
pub fn simplify_line_strings(
    array: &LineStringArray,
    tolerance: f64,
    preserve_topology: bool,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<LineStringArray> {
    let geos_geometries = simplify_via_geos(
        array.to_geos_rows(on_row_error, early_coord_check)?,
        tolerance,
        preserve_topology,
        on_row_error,
    )?;
    with_const_flags!(
        Z = array.coords.dim().has_z();
        geos_geometries.into_line_string_array::<Z>(on_row_error)
    )
}

/// Like [`simplify_line_strings`], for multilinestrings.
pub fn simplify_multi_line_strings(
    array: &MultiLineStringArray,
    tolerance: f64,
    preserve_topology: bool,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<MultiLineStringArray> {
    let geos_geometries = simplify_via_geos(
        array.to_geos_rows(on_row_error, early_coord_check)?,
        tolerance,
        preserve_topology,
        on_row_error,
    )?;
    with_const_flags!(
        Z = array.coords.dim().has_z();
        geos_geometries.into_multi_line_string_array::<Z>(on_row_error)
    )
}

/// Like [`simplify_line_strings`], for polygons.
/// A polygon that GEOS splits into several fails.
pub fn simplify_polygons(
    array: &PolygonArray,
    tolerance: f64,
    preserve_topology: bool,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<PolygonArray> {
    simplify_via_geos(
        array.to_geos_rows(on_row_error, early_coord_check)?,
        tolerance,
        preserve_topology,
        on_row_error,
    )?
    .into_polygon_array(
        array.coords.dim().has_z(),
        on_row_error,
        NewRingOrientation::Keep,
        NewHoles::Opposite,
    )
}

/// Like [`simplify_line_strings`], for multipolygons.
pub fn simplify_multi_polygons(
    array: &MultiPolygonArray,
    tolerance: f64,
    preserve_topology: bool,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<MultiPolygonArray> {
    simplify_via_geos(
        array.to_geos_rows(on_row_error, early_coord_check)?,
        tolerance,
        preserve_topology,
        on_row_error,
    )?
    .into_multi_polygon_array(
        array.coords.dim().has_z(),
        on_row_error,
        NewRingOrientation::Keep,
        NewHoles::Opposite,
    )
}

/// Simplifies every geometry to within `tolerance` through GEOS.
/// A row GEOS fails on goes to `on_row_error`.
fn simplify_via_geos(
    mut geometries: GeosGeometries,
    tolerance: f64,
    preserve_topology: bool,
    on_row_error: OnRowError,
) -> GeoComputeResult<GeosGeometries> {
    with_const_flags!(
        PRESERVE_TOPOLOGY = preserve_topology;
        simplify_via_geos_with::<PRESERVE_TOPOLOGY>(
            &mut geometries.geometries,
            tolerance,
            on_row_error,
        )
    )?;
    // The size hint is kept as an upper bound, as simplifying removes vertices.
    // Only where GEOS repairs a crossing can it add some.
    Ok(geometries)
}

/// See [`simplify_via_geos`].
/// Errors are the exception, so `on_row_error` is checked per row rather than made const.
fn simplify_via_geos_with<const PRESERVE_TOPOLOGY: bool>(
    geometries: &mut GeosRows,
    tolerance: f64,
    on_row_error: OnRowError,
) -> GeoComputeResult<()> {
    geometries.replace_geometries(on_row_error, |_, input| {
        let simplified = if PRESERVE_TOPOLOGY {
            input.topology_preserve_simplify(tolerance)
        } else {
            // TODO: GEOS keeps m here, but the `into_*_array` conversions drop it.
            input.simplify(tolerance)
        };
        Ok(simplified?)
    })
}
