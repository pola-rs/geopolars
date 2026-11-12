//! Interfacing with the `rsgeo` crate (the Rust `geo` crate).
//!
//! rsgeo has geometry types of its own, so what it relates is copied into them.
//! Its algorithms only look at x and y, so z and m are never read.

#[cfg(feature = "native")]
#[allow(dead_code)]
mod is_valid;
// Unused: rings_overlap runs through GEOS.
#[allow(dead_code)]
mod rings_overlap;
// Unused: self_intersects runs through GEOS.
#[allow(dead_code)]
mod self_intersects;

#[cfg(feature = "native")]
#[allow(unused_imports)]
pub(crate) use is_valid::{multi_polygon_topology_is_valid, polygon_topology_is_valid};
#[cfg(feature = "native")]
pub use is_valid::{MultiPolygonTopologyChecks, PolygonTopologyChecks};
pub use rings_overlap::{rings_overlap_against_one, rings_overlap_pairwise};
pub use self_intersects::{
    line_strings_self_intersect, multi_line_strings_self_intersect, multi_polygons_self_intersect,
    polygons_self_intersect, vertices_self_intersect,
};

use ::rsgeo::{Coord, LineString, Polygon};
use geopolars_views::{is_ring, ScalarLineString};

use crate::per_row::{per_part_optional_bool, per_row_optional_bool};
use crate::{GeoComputeError, GeoComputeResult};

/// Whether every vertex of `line` is finite in x and y.
#[inline(always)]
fn xy_is_finite(line: &ScalarLineString) -> bool {
    let x = &line.coords.x()[line.vertices.clone()];
    let y = &line.coords.y()[line.vertices.clone()];
    x.iter().chain(y).all(|value| value.is_finite())
}

/// Whether `line` is finite in x and y.
/// Errors if it is, but is not a ring.
#[inline(always)]
fn ring_is_finite(line: &ScalarLineString) -> GeoComputeResult<bool> {
    if !xy_is_finite(line) {
        return Ok(false);
    }
    if !is_ring::<false>(line.coords, line.vertices.clone()) {
        return Err(GeoComputeError::NotARing {
            vertices: line.vertices.len(),
        });
    }
    Ok(true)
}

/// The polygon the ring `line` encloses, in x and y.
/// This is the only copy made of its vertices.
fn to_rsgeo_ring(line: &ScalarLineString) -> Polygon<f64> {
    let vertices = line
        .positions::<false>()
        .map(|[x, y, _]| Coord { x, y })
        .collect();
    Polygon::new(LineString::new(vertices), vec![])
}
