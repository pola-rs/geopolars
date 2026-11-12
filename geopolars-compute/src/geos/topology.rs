//! Topology checks used in `is_valid_ogc`, as predicates on one polygon or multipolygon.
//! Each predicate stops at the first failure.
//! Rings have to be finite in x and y to be related.

use geopolars_views::ScalarPolygon;
use geos::{Geometry, PreparedGeometry};

use super::axes;
use super::polygons::polygon_enclosed_by_ring_to_geos;
use crate::GeoComputeResult;

/// A hole inside the exterior, their boundaries meeting at points.
/// The case where they don't meet at all is `contains_properly`.
const INSIDE_TOUCHING_AT_POINTS: &str = "T***0*FF*";

/// Two polygons sharing no area, their boundaries meeting at points.
/// The case where they don't meet at all is `disjoint`.
const TOUCHING_AT_POINTS: &str = "F***0****";

pub fn meet_only_at_points_and_no_overlap(polygons: &[Geometry]) -> GeoComputeResult<bool> {
    for (i, polygon) in polygons.iter().enumerate() {
        let polygon = PreparedGeometry::new(polygon)?;
        for other in &polygons[i + 1..] {
            if !(polygon.disjoint(other)? || polygon.relate_pattern(other, TOUCHING_AT_POINTS)?) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Whether every hole of `polygon` lies inside the exterior,
/// and no two rings overlap or touch along a line.
/// Each ring is related as the polygon it encloses.
pub fn polygon_rings_fit(polygon: &ScalarPolygon) -> GeoComputeResult<bool> {
    let axes = axes(polygon.coords);
    let rings = polygon
        .parts
        .clone()
        .map(|j| polygon_enclosed_by_ring_to_geos(&axes, polygon.rings.range(j)))
        .collect::<GeoComputeResult<Vec<_>>>()?;
    let [exterior, holes @ ..] = rings.as_slice() else {
        return Ok(true);
    };
    let exterior = PreparedGeometry::new(exterior)?;
    for hole in holes {
        if !(exterior.contains_properly(hole)?
            || exterior.relate_pattern(hole, INSIDE_TOUCHING_AT_POINTS)?)
        {
            return Ok(false);
        }
    }
    meet_only_at_points_and_no_overlap(holes)
}
