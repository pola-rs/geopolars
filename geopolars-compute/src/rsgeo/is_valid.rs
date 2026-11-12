//! Topology checks used in `is_valid_ogc`, as predicates on one polygon or multipolygon.
//! Answers how the rings of a polygon, and the parts of a multipolygon, fit together in x and y.
//! See `is_valid_ogc` in Python for the rules.
//!
//! Every polygon is assumed to have a valid shape already (all rings empty, or all of them rings).
//! The typed views of [crate::ogc_validity] stand for that.
//! Rings have to be finite in x and y to be related, so that is checked here first.

use ::rsgeo::coordinate_position::CoordPos;
use ::rsgeo::dimensions::Dimensions;
use ::rsgeo::relate::IntersectionMatrix;
use ::rsgeo::{Polygon, PreparedGeometry, Relate};
use geopolars_views::{ScalarLineString, ScalarMultiPolygon, ScalarPolygon};

use super::rings_overlap::overlap;
use super::self_intersects::finite_line_self_intersects;
use super::{to_rsgeo_ring, xy_is_finite};

fn touch_only_at_points(matrix: &IntersectionMatrix) -> bool {
    matrix.get(CoordPos::OnBoundary, CoordPos::OnBoundary) != Dimensions::OneDimensional
}

/// Whether every hole lies inside the exterior, and no two rings overlap or touch along a line.
/// Each ring is given as the polygon it encloses.
/// The first is the exterior, the others are holes.
fn holes_lie_inside_exterior_without_overlap(rings: &[Polygon<f64>]) -> bool {
    // Re-use matrix, as getting that is the expensive part.
    let [exterior, holes @ ..] = rings else {
        return true;
    };
    let exterior = PreparedGeometry::from(exterior);
    let holes: Vec<_> = holes.iter().map(PreparedGeometry::from).collect();
    holes.iter().enumerate().all(|(i, hole)| {
        let matrix = exterior.relate(hole);
        matrix.is_contains()
            && touch_only_at_points(&matrix)
            && holes[i + 1..].iter().all(|other| {
                let matrix = hole.relate(other);
                !overlap::<false>(&matrix) && touch_only_at_points(&matrix)
            })
    })
}

/// Whether no two parts of a multipolygon share any area, and they only touch at points.
fn parts_do_not_overlap(parts: &[Polygon<f64>]) -> bool {
    let prepared: Vec<_> = parts.iter().map(PreparedGeometry::from).collect();
    prepared.iter().enumerate().all(|(i, part)| {
        prepared[i + 1..].iter().all(|other| {
            let matrix = part.relate(other);
            !overlap::<false>(&matrix) && touch_only_at_points(&matrix)
        })
    })
}

/// The topology checks to run on a polygon. See `is_valid_ogc` in Python.
#[derive(Debug, Clone, Copy)]
pub struct PolygonTopologyChecks {
    pub check_ring_self_intersect: bool,
    pub check_polygon_self_overlap: bool,
}

impl PolygonTopologyChecks {
    pub fn checks_anything(&self) -> bool {
        self.check_ring_self_intersect || self.check_polygon_self_overlap
    }
}

/// The topology checks to run on a multipolygon. See `is_valid_ogc` in Python.
#[derive(Debug, Clone, Copy)]
pub struct MultiPolygonTopologyChecks {
    /// What every part is checked for.
    pub polygons: PolygonTopologyChecks,
    pub check_multi_polygon_self_overlap: bool,
}

impl MultiPolygonTopologyChecks {
    pub fn checks_anything(&self) -> bool {
        self.polygons.checks_anything() || self.check_multi_polygon_self_overlap
    }
}

/// Whether `ring` does not cross itself, if `check_ring_self_intersect=true`.
fn ring_passes_self_intersect_check(
    ring: &ScalarLineString,
    checks: &PolygonTopologyChecks,
) -> bool {
    !(checks.check_ring_self_intersect && finite_line_self_intersects(ring))
}

/// Runs the polygon checks of `checks` on `polygon`.
/// If `keep=true`, its rings are returned as one rsgeo polygon to relate further.
/// `Err(())` if a check fails or a ring is not finite in x and y,
/// `Ok(None)` for the empty polygon (or without `keep`).
fn check_polygon_topology(
    polygon: &ScalarPolygon,
    checks: &PolygonTopologyChecks,
    keep: bool,
) -> Result<Option<Polygon<f64>>, ()> {
    // A valid shape has either every ring empty, or none.
    if polygon.has_empty_exterior() {
        return Ok(None);
    }
    if !polygon.rings().all(|ring| xy_is_finite(&ring)) {
        return Err(());
    }
    if !polygon
        .rings()
        .all(|ring| ring_passes_self_intersect_check(&ring, checks))
    {
        return Err(());
    }
    // A lone exterior has nothing to fit with.
    let fit = checks.check_polygon_self_overlap && polygon.parts.len() > 1;
    if !(fit || keep) {
        return Ok(None);
    }
    // Handed to rsgeo, which needs them owned: the only copy made of the vertices.
    let rings: Vec<_> = polygon.rings().map(|ring| to_rsgeo_ring(&ring)).collect();
    if fit && !holes_lie_inside_exterior_without_overlap(&rings) {
        return Err(());
    }
    Ok(keep.then(|| {
        let mut rings = rings.into_iter().map(|ring| ring.into_inner().0);
        let exterior = rings
            .next()
            .expect("a polygon that is not empty has an exterior");
        Polygon::new(exterior, rings.collect())
    }))
}

pub(crate) fn polygon_topology_is_valid(
    polygon: &ScalarPolygon,
    checks: &PolygonTopologyChecks,
) -> bool {
    check_polygon_topology(polygon, checks, false).is_ok()
}

/// [polygon_topology_is_valid] for every part, and,
/// if `check_multi_polygon_self_overlap=true`, whether the [parts do not overlap](parts_do_not_overlap).
pub(crate) fn multi_polygon_topology_is_valid(
    multi: &ScalarMultiPolygon,
    checks: &MultiPolygonTopologyChecks,
) -> bool {
    let check_parts = checks.check_multi_polygon_self_overlap;
    let mut parts = Vec::new();
    for polygon in multi.polygons() {
        match check_polygon_topology(&polygon, &checks.polygons, check_parts) {
            Ok(part) => parts.extend(part),
            Err(()) => return false,
        }
    }
    !check_parts || parts_do_not_overlap(&parts)
}
