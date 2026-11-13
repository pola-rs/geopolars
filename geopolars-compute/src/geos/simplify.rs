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
//!
//! ## `EARLY_FINITE_CHECK`
//!
//! The public functions take `early_finite_check`.
//! See [super] for `EARLY_FINITE_CHECK`.

use geopolars_views::{LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray};

use super::{GeosGeometries, GeosRow, ToGeosRows};
use crate::GeoComputeResult;

/// Simplifies every linestring to within `tolerance` through GEOS.
pub fn simplify_line_strings(
    array: &LineStringArray,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<LineStringArray> {
    let geos_geometries = simplify_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
    )?;
    with_const_flags!(
        Z = array.coords.dim().has_z();
        geos_geometries.into_line_string_array::<Z>(ignore_per_row_errors)
    )
}

/// Like [`simplify_line_strings`], for multilinestrings.
pub fn simplify_multi_line_strings(
    array: &MultiLineStringArray,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<MultiLineStringArray> {
    let geos_geometries = simplify_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
    )?;
    with_const_flags!(
        Z = array.coords.dim().has_z();
        geos_geometries.into_multi_line_string_array::<Z>(ignore_per_row_errors)
    )
}

/// Like [`simplify_line_strings`], for polygons.
/// A polygon that GEOS splits into several fails.
pub fn simplify_polygons(
    array: &PolygonArray,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<PolygonArray> {
    simplify_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
    )?
    .into_polygon_array(array.coords.dim().has_z(), ignore_per_row_errors, false)
}

/// Like [`simplify_line_strings`], for multipolygons.
pub fn simplify_multi_polygons(
    array: &MultiPolygonArray,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<MultiPolygonArray> {
    simplify_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
    )?
    .into_multi_polygon_array(array.coords.dim().has_z(), ignore_per_row_errors, false)
}

/// Every geometry simplified to within `tolerance`, forwarded to GEOS.
/// A null or empty geometry stays as it is.
/// If `ignore_per_row_errors=true`, a geometry GEOS fails on becomes null;
/// otherwise its error is returned.
/// The size hint is kept as an upper bound: simplifying removes vertices.
/// Only where GEOS repairs a crossing can it add some.
fn simplify_via_geos(
    mut geometries: GeosGeometries,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<GeosGeometries> {
    with_const_flags!(
        PRESERVE_TOPOLOGY = preserve_topology;
        simplify_via_geos_with::<PRESERVE_TOPOLOGY>(
            &mut geometries.geometries,
            tolerance,
            ignore_per_row_errors,
        )
    )?;
    Ok(geometries)
}

/// See [`simplify_via_geos`].
/// Errors are the exception, so `ignore_per_row_errors` is checked per row rather than made const.
fn simplify_via_geos_with<const PRESERVE_TOPOLOGY: bool>(
    geometries: &mut [GeosRow],
    tolerance: f64,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<()> {
    for geometry in geometries {
        let GeosRow::Geometry(input) = geometry else {
            continue;
        };
        let simplified = if PRESERVE_TOPOLOGY {
            input.topology_preserve_simplify(tolerance)
        } else {
            // TODO: GEOS keeps m here, but the `into_*_array` conversions drop it.
            input.simplify(tolerance)
        };
        *geometry = match simplified {
            Ok(simple) => GeosRow::Geometry(simple),
            Err(e) if !ignore_per_row_errors => return Err(e.into()),
            Err(_) => GeosRow::Null,
        };
    }
    Ok(())
}
