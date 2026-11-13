//! Whether each geometry is valid by OGC Simple Features.
//! Implements the Python `is_valid_ogc`.
//!
//! Every geometry is held to the checks of [crate::geoarrow_validity] first,
//! with every coordinate finite as the Python `is_valid_ogc` asks.
//! Its shape is checked natively too, with a mask per level like those checks.
//! Then GEOS checks its topology.
//! It only does so for the rows that pass every other check.
//! Each kernel gives a `bool` per row, `false` for a null row.
//! The polygon kernels error when GEOS does.
//! With every topology check on, GEOS's `is_valid` runs them all at once.
//! Otherwise, each check runs on its own.
//!
//! Null values below the rows are not looked at here, as in [crate::geoarrow_validity].

use geopolars_views::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, ScalarMultiPolygon, ScalarPolygon,
};
use geos::{Geom, Geometry};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and;

use crate::geoarrow_validity::{multi_points_are_valid_geoarrow, points_are_valid_geoarrow};
use crate::geos::{meet_only_at_points_and_no_overlap, polygon_rings_fit, ToGeos};
use crate::native::{
    line_shapes_are_valid, polygon_shapes_are_valid, rings_are_closed, which_are_not_selected,
    Bounds, CoordinateSelection, NonFinite,
};
use crate::passing::{passing_bool_array, rows_passing};
use crate::GeoComputeResult;

/// The topology checks of the Python `is_valid_ogc` that apply to a polygon.
#[derive(Debug, Clone, Copy)]
pub struct PolygonTopologyChecks {
    pub check_ring_self_intersect: bool,
    pub check_polygon_self_overlap: bool,
}

impl PolygonTopologyChecks {
    pub fn checks_anything(&self) -> bool {
        self.check_ring_self_intersect || self.check_polygon_self_overlap
    }

    pub fn checks_everything(&self) -> bool {
        self.check_ring_self_intersect && self.check_polygon_self_overlap
    }
}

/// The topology checks of the Python `is_valid_ogc`.
#[derive(Debug, Clone, Copy)]
pub struct MultiPolygonTopologyChecks {
    /// Every part is checked for these individually.
    pub polygons: PolygonTopologyChecks,
    pub check_multi_polygon_self_overlap: bool,
}

impl MultiPolygonTopologyChecks {
    pub fn checks_anything(&self) -> bool {
        self.polygons.checks_anything() || self.check_multi_polygon_self_overlap
    }

    pub fn checks_everything(&self) -> bool {
        self.polygons.checks_everything() && self.check_multi_polygon_self_overlap
    }
}

/// Selects every coordinate that is not finite, or lies outside `bounds`.
/// With `select_empty=true`, the empty point is selected too.
fn finite_selection(bounds: Option<Bounds>, select_empty: bool) -> CoordinateSelection {
    CoordinateSelection {
        non_finite: Some(NonFinite {
            select_non_finite: true,
            select_empty,
            ignore_z: false,
            ignore_m: false,
        }),
        bounds,
    }
}

fn needs_whole_polygon_in_geos(checks: &PolygonTopologyChecks) -> bool {
    checks.checks_everything() || checks.check_ring_self_intersect
}

fn whole_polygon_passes(
    geometry: &Geometry,
    checks: &PolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    Ok(if checks.checks_everything() {
        geometry.is_valid()?
    } else {
        !checks.check_ring_self_intersect || geometry.is_simple()?
    })
}

fn rings_fit_if_checked_separately(
    polygon: &ScalarPolygon,
    checks: &PolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    // A lone exterior has nothing to fit with.
    let fit = checks.check_polygon_self_overlap && !checks.checks_everything();
    Ok(!fit || polygon.parts.len() < 2 || polygon_rings_fit(polygon)?)
}

/// Whether `polygon` passes the topology `checks`.
/// It has to have a valid shape, and be finite in x and y.
fn polygon_topology_is_valid(
    polygon: &ScalarPolygon,
    checks: &PolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    if polygon.has_empty_exterior() {
        return Ok(true);
    }
    if needs_whole_polygon_in_geos(checks) && !whole_polygon_passes(&polygon.to_geos()?, checks)? {
        return Ok(false);
    }
    rings_fit_if_checked_separately(polygon, checks)
}

/// Uses [polygon_topology_is_valid] for every part.
/// if `check_multi_polygon_self_overlap=true`, uses [meet_only_at_points_and_no_overlap] for its parts.
fn multi_polygon_topology_is_valid(
    multi: &ScalarMultiPolygon,
    checks: &MultiPolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    if checks.checks_everything() {
        return Ok(multi.to_geos()?.is_valid()?);
    }
    let polygons = multi
        .polygons()
        .filter(|polygon| !polygon.has_empty_exterior());
    if !checks.check_multi_polygon_self_overlap {
        for polygon in polygons {
            if !polygon_topology_is_valid(&polygon, &checks.polygons)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    // Every part is built once, for its own checks as well as to relate to the others.
    let mut parts = Vec::with_capacity(multi.parts.len());
    for polygon in polygons {
        let geometry = polygon.to_geos()?;
        if !whole_polygon_passes(&geometry, &checks.polygons)?
            || !rings_fit_if_checked_separately(&polygon, &checks.polygons)?
        {
            return Ok(false);
        }
        parts.push(geometry);
    }
    meet_only_at_points_and_no_overlap(&parts)
}

// TODO: Not yet converted to bitmask pass, but is a minor candidate for it.
// Only visit the rows set in `passing`, and give `None` when every row passes.
// GEOS costs far more per row than the loop does.
/// `passing`, without the rows where `test` fails.
/// `test` only runs on the rows set in `passing`.
fn try_and_where_passing<E>(
    len: usize,
    passing: Option<Bitmap>,
    test: impl Fn(usize) -> Result<bool, E>,
) -> Result<Option<Bitmap>, E> {
    let is_passing = |i: usize| passing.as_ref().is_none_or(|passing| passing.get_bit(i));
    let mut tested = MutableBitmap::with_capacity(len);
    for i in 0..len {
        tested.push(is_passing(i) && test(i)?);
    }
    Ok(Some(tested.into()))
}

pub fn points_are_valid_ogc(array: &PointArray, bounds: Option<Bounds>) -> BooleanArray {
    points_are_valid_geoarrow(array, &finite_selection(bounds, false))
}

pub fn multi_points_are_valid_ogc(array: &MultiPointArray, bounds: Option<Bounds>) -> BooleanArray {
    multi_points_are_valid_geoarrow(array, &finite_selection(bounds, false))
}

pub fn line_strings_are_valid_ogc(array: &LineStringArray, bounds: Option<Bounds>) -> BooleanArray {
    let (rows, coords) = (&array.rows, &array.coords);
    let shapes = line_shapes_are_valid(rows.ignoring_validity(), coords);
    let passing = rows_passing(
        &[rows.ignoring_validity()],
        vec![
            combine_validities_and(rows.validity(), Some(&shapes)),
            which_are_not_selected(coords, &finite_selection(bounds, true)),
        ],
    );
    passing_bool_array(rows.len(), rows.validity(), passing)
}

pub fn multi_line_strings_are_valid_ogc(
    array: &MultiLineStringArray,
    bounds: Option<Bounds>,
) -> BooleanArray {
    let (rows, lines, coords) = (&array.rows, &array.lines, &array.coords);
    let passing = rows_passing(
        &[rows.ignoring_validity(), lines],
        vec![
            rows.validity().cloned(),
            Some(line_shapes_are_valid(lines, coords)),
            which_are_not_selected(coords, &finite_selection(bounds, true)),
        ],
    );
    passing_bool_array(rows.len(), rows.validity(), passing)
}

pub fn polygons_are_valid_ogc(
    array: &PolygonArray,
    bounds: Option<Bounds>,
    topology: &PolygonTopologyChecks,
) -> GeoComputeResult<BooleanArray> {
    let (rows, rings, coords) = (&array.rows, &array.rings, &array.coords);
    let shapes = polygon_shapes_are_valid(rows.ignoring_validity(), rings, coords);
    let mut passing = rows_passing(
        &[rows.ignoring_validity(), rings],
        vec![
            combine_validities_and(rows.validity(), Some(&shapes)),
            Some(rings_are_closed(rings, coords)),
            which_are_not_selected(coords, &finite_selection(bounds, true)),
        ],
    );
    if topology.checks_anything() {
        passing = try_and_where_passing(rows.len(), passing, |i| {
            polygon_topology_is_valid(&array.get(i), topology)
        })?;
    }
    Ok(passing_bool_array(rows.len(), rows.validity(), passing))
}

pub fn multi_polygons_are_valid_ogc(
    array: &MultiPolygonArray,
    bounds: Option<Bounds>,
    topology: &MultiPolygonTopologyChecks,
) -> GeoComputeResult<BooleanArray> {
    let (rows, polygons, rings, coords) =
        (&array.rows, &array.polygons, &array.rings, &array.coords);
    let mut passing = rows_passing(
        &[rows.ignoring_validity(), polygons, rings],
        vec![
            rows.validity().cloned(),
            Some(polygon_shapes_are_valid(polygons, rings, coords)),
            Some(rings_are_closed(rings, coords)),
            which_are_not_selected(coords, &finite_selection(bounds, true)),
        ],
    );
    if topology.checks_anything() {
        passing = try_and_where_passing(rows.len(), passing, |i| {
            multi_polygon_topology_is_valid(&array.get(i), topology)
        })?;
    }
    Ok(passing_bool_array(rows.len(), rows.validity(), passing))
}
