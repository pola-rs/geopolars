//! Convex hulls, forwarded to GEOS.
//!
//! A hull only depends on the points, not on the kind of geometry.
//! We use this to our advantage, and make every row go to GEOS the same way:
//! as one linestring (through its coordinates).
//! For a polygon or a multipolygon, these are only the coordinates of the exteriors.
//! A valid polygon will not change in size if you add or remove interior sections after all.
//!
//! An empty row, or a row of one coordinate, never goes to GEOS,
//! as its hull is known in advance.
//!
//! This trick almost always gives the same hull as GEOS on the geometry itself,
//! but there are some edge cases:
//! - GEOS ignores the holes of a polygon, but not those of a multipolygon.
//!   Such a multipolygon is invalid, so we take it to be a GEOS bug, and ignore holes for both.
//! - GEOS switches algorithm above 50 coordinates (`TUNING_REDUCE_SIZE`),
//!   counting duplicates and the closing vertex.
//!   Holes add to that count for a multipolygon in GEOS, but not here.
//!   So near that boundary, one of the two can take the other path.
//!   Then nearly collinear input can give a different (sometimes non-convex) sliver hull,
//!   and NaN input an error instead of a hull, or the other way around.
//! - No rings are built.
//!   GEOS can't build a ring starting with NaN, because its closing NaN ≠ NaN.
//!   Here, that ring is hulled like any other NaN input.
//!
//! # Const generic flags
//!
//! See [super] for `EARLY_COORD_CHECK`, `ALL_VALID` and `Z`.
//! With [EarlyCoordCheck::Off], what happens to NaN coordinates depends on where they are placed.
//! [EarlyCoordCheck::Finite] only lets empty points (x and y NaN) through,
//! so the first three cases below apply to it too.
//! - only empty points -> a `POINT (NaN NaN)`, so the row is null
//! - an empty point first, then real ones -> also a `POINT (NaN NaN)`,
//!   so the row is silently null, even with `on_row_error=`[OnRowError::Raise];
//! - an empty point after a real one -> it is left out, giving the hull of the others
//! - a vertex with just one NaN axis -> an error ("orientationIndex encountered NaN/Inf numbers").
//!   So the error is returned, or with `on_row_error=`[OnRowError::Null] the row is null.
//!
//! With [EarlyCoordCheck::FiniteAndEmpty], each of these rows fails the same way a GEOS error does,
//! which makes this consistent.

use std::ops::Range;

use geopolars_views::output::SizeHint;
use geopolars_views::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
    ValidCoords,
};
use geos::{CoordSeq, Geom, Geometry, GeometryTypes};

use super::{
    axes, coords_pass_early_check, GeometryOrientation, GeosGeometries, GeosRow, GeosRows,
};
use crate::{
    row_or_null, EarlyCoordCheck, GeoComputeResult, NewHoles, NewRingOrientation, OnRowError,
};

// TODO: a geometry crossing the antimeridian is hulled the long way round.
// We could add a flag that checks if the input crosses it (>180°),
// and then reproject them back.

/// Finds the convex hull of every linestring through GEOS.
/// Implements the Python `convex_hull` for linestrings, which documents its behaviour.
pub fn line_string_convex_hulls(
    array: &LineStringArray,
    on_row_error: OnRowError,
    new_ring_orientation: NewRingOrientation,
    new_holes: NewHoles,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<PolygonArray> {
    hulls(
        &array.rows,
        &array.coords,
        on_row_error,
        new_ring_orientation,
        new_holes,
        early_coord_check,
        |vertices, ranges| ranges.push(vertices),
    )
}

/// Like [`line_string_convex_hulls`], with one hull around all the points.
pub fn multi_point_convex_hulls(
    array: &MultiPointArray,
    on_row_error: OnRowError,
    new_ring_orientation: NewRingOrientation,
    new_holes: NewHoles,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<PolygonArray> {
    hulls(
        &array.rows,
        &array.coords,
        on_row_error,
        new_ring_orientation,
        new_holes,
        early_coord_check,
        |vertices, ranges| ranges.push(vertices),
    )
}

/// Like [`line_string_convex_hulls`], with one hull of all the parts as a whole.
pub fn multi_line_string_convex_hulls(
    array: &MultiLineStringArray,
    on_row_error: OnRowError,
    new_ring_orientation: NewRingOrientation,
    new_holes: NewHoles,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<PolygonArray> {
    hulls(
        &array.rows,
        &array.coords,
        on_row_error,
        new_ring_orientation,
        new_holes,
        early_coord_check,
        |parts, ranges| ranges.extend(parts.map(|j| array.lines.range(j))),
    )
}

/// Like [`line_string_convex_hulls`], hulling only the exterior.
pub fn polygon_convex_hulls(
    array: &PolygonArray,
    on_row_error: OnRowError,
    new_ring_orientation: NewRingOrientation,
    new_holes: NewHoles,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<PolygonArray> {
    hulls(
        &array.rows,
        &array.coords,
        on_row_error,
        new_ring_orientation,
        new_holes,
        early_coord_check,
        |parts, ranges| {
            if !parts.is_empty() {
                ranges.push(array.rings.range(parts.start));
            }
        },
    )
}

/// Like [`line_string_convex_hulls`],
/// with one hull of the exteriors of all the parts as a whole.
pub fn multi_polygon_convex_hulls(
    array: &MultiPolygonArray,
    on_row_error: OnRowError,
    new_ring_orientation: NewRingOrientation,
    new_holes: NewHoles,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<PolygonArray> {
    hulls(
        &array.rows,
        &array.coords,
        on_row_error,
        new_ring_orientation,
        new_holes,
        early_coord_check,
        |parts, ranges| {
            ranges.extend(
                parts
                    .map(|p| array.polygons.range(p))
                    .filter(|rings| !rings.is_empty())
                    .map(|rings| array.rings.range(rings.start)),
            )
        },
    )
}

/// `coord_ranges` adds the ranges of `coords` to hull for the parts of a row.
fn hulls(
    rows: &List,
    coords: &ValidCoords,
    on_row_error: OnRowError,
    new_ring_orientation: NewRingOrientation,
    new_holes: NewHoles,
    early_coord_check: EarlyCoordCheck,
    coord_ranges: impl Fn(Range<usize>, &mut Vec<Range<usize>>),
) -> GeoComputeResult<PolygonArray> {
    let has_z = coords.dim().has_z();
    let hulls = with_const_flags!(
        Z = has_z,
        ALL_VALID = rows.is_all_valid(),
        EARLY_COORD_CHECK: EarlyCoordCheck = early_coord_check;
        hulls_with::<Z, ALL_VALID, EARLY_COORD_CHECK>(
            rows,
            coords,
            on_row_error,
            coord_ranges,
        )
    )?;
    hulls.into_polygon_array(has_z, on_row_error, new_ring_orientation, new_holes)
}

/// Builds the convex hull of every row as a GEOS polygon, for [`hulls`].
fn hulls_with<const Z: bool, const ALL_VALID: bool, const EARLY_COORD_CHECK: u8>(
    rows: &List,
    coords: &ValidCoords,
    on_row_error: OnRowError,
    coord_ranges: impl Fn(Range<usize>, &mut Vec<Range<usize>>),
) -> GeoComputeResult<GeosGeometries> {
    let (in_x, in_y, in_z, _) = axes(coords);
    let in_z = in_z.unwrap_or_default();

    // The coordinates of one row, gathered from its ranges, and reused across rows.
    // GEOS drops m, so it isn't copied.
    let mut ranges = Vec::new();
    let mut row_x = Vec::new();
    let mut row_y = Vec::new();
    let mut row_z = Vec::new();

    // `None` if it is not a polygon (the hull collapsed)
    let hull = |row_x: &[f64], row_y: &[f64], row_z: &[f64]| -> GeoComputeResult<_> {
        coords_pass_early_check::<EARLY_COORD_CHECK>(row_x, row_y)?;
        let seq = CoordSeq::new_from_arrays(row_x, row_y, Z.then_some(row_z), None)?;
        let hull = Geometry::create_line_string(seq)?.convex_hull()?;
        debug_assert!(!hull.is_empty()?, "Check before offset should catch this");
        if hull.geometry_type()? != GeometryTypes::Polygon {
            return Ok(None);
        }
        Ok(Some(hull))
    };

    let mut hulls = GeosRows::with_capacity(rows.len());
    for i in 0..rows.len() {
        // A null row's offsets point anywhere, so it is not converted.
        if !ALL_VALID && !rows.is_valid(i) {
            hulls.push(GeosRow::Null);
            continue;
        }
        ranges.clear();
        coord_ranges(rows.range(i), &mut ranges);
        row_x.clear();
        row_y.clear();
        row_z.clear();
        for range in &ranges {
            row_x.extend_from_slice(&in_x[range.clone()]);
            row_y.extend_from_slice(&in_y[range.clone()]);
            if Z {
                row_z.extend_from_slice(&in_z[range.clone()]);
            }
        }
        hulls.push(match row_x.len() {
            // An empty row has an empty hull, without a trip to GEOS.
            0 => GeosRow::Empty,
            // A single coordinate encloses nothing, and GEOS can't build a linestring of it.
            // The Python `convex_hull` documents its null polygon, so it is not a failure.
            1 => GeosRow::Null,
            _ => match row_or_null(hull(&row_x, &row_y, &row_z).map(Some), on_row_error)? {
                Some(Some(hull)) => GeosRow::Geometry {
                    geometry: hull,
                    maybe_wrong_orientation: false,
                },
                // Like a single coordinate, a hull that encloses nothing is null.
                // A row that failed under `on_row_error="null"` is null as well.
                _ => GeosRow::Null,
            },
        });
    }
    Ok(GeosGeometries {
        geometries: hulls,
        // An upper bound, as a row's hull is one ring,
        // which closes on at most one coordinate more than the row has.
        size_hint: SizeHint {
            coords: coords.len() + rows.len(),
            parts: [rows.len(), 0],
        },
        // GEOS builds every hull anew, from a linestring of the row's coordinates.
        orientation: GeometryOrientation::GEOS_BUILT,
    })
}
