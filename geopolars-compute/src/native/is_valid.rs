//! The ring and shape checks of `is_valid_geoarrow` and `is_valid_ogc`.
//! See them in Python for the rules.
//! The coordinate checks are in [super::coordinate_selection].
//! Each check gives a bitmap over a whole level, set where an item passes.
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! The ring closure is generic over `Z`.
//! If `Z=true`, a ring has to close in z as well.

use std::ops::Range;

use geopolars_core::{same_place, ValidCoords, ValidList};
#[cfg(feature = "geos")]
use geopolars_core::{ScalarPolygon, MIN_RING_VERTICES};
use polars_arrow::bitmap::Bitmap;

/// Whether `ring` is empty or closed, as GeoArrow asks.
/// Unlike [is_ring](geopolars_core::is_ring), this asks for no minimum number of vertices.
#[inline(always)]
pub(crate) fn is_closed_or_empty<const Z: bool>(coords: &ValidCoords, ring: Range<usize>) -> bool {
    ring.is_empty() || same_place::<Z>(coords, ring.start, ring.end - 1)
}

fn rings_are_closed_with<const Z: bool>(rings: &ValidList, coords: &ValidCoords) -> Bitmap {
    Bitmap::from_iter(
        rings
            .ranges()
            .map(|ring| is_closed_or_empty::<Z>(coords, ring)),
    )
}

/// Whether each of `rings` passes [is_closed_or_empty].
pub(crate) fn rings_are_closed(rings: &ValidList, coords: &ValidCoords) -> Bitmap {
    let z = coords.dim().has_z();
    with_const_flags!(Z = z; rings_are_closed_with::<Z>(rings, coords))
}

/// Whether `vertices` hold at least `min` vertices that lie elsewhere in x and y than the one before.
#[cfg(feature = "geos")]
#[inline(always)]
fn has_at_least_distinct_vertices(
    coords: &ValidCoords,
    vertices: Range<usize>,
    min: usize,
) -> bool {
    let start = vertices.start;
    vertices
        .filter(|&i| i == start || !same_place::<false>(coords, i - 1, i))
        .take(min)
        .count()
        == min
}

/// OGC accepts lines with at least two distinct vertices (or empty)
#[cfg(feature = "geos")]
pub(crate) fn line_shapes_are_valid(lines: &ValidList, coords: &ValidCoords) -> Bitmap {
    Bitmap::from_iter(
        lines
            .ranges()
            .map(|line| line.is_empty() || has_at_least_distinct_vertices(coords, line, 2)),
    )
}

// TODO: Not yet converted to bitmask pass, but is a good candidate for it.
// Mark the rings that are empty or have at least 4 distinct vertices in a mask,
// and fold it with `rows_passing`.
// The polygons then only need a mask from the offsets: either every ring is empty, or none is.
/// Whether `polygon` has a shape OGC accepts:
/// every ring empty, or every ring with at least 4 distinct vertices.
#[cfg(feature = "geos")]
#[inline(always)]
fn polygon_shape_is_valid(polygon: &ScalarPolygon) -> bool {
    if polygon.has_empty_exterior() {
        return polygon.rings().all(|ring| ring.vertices.is_empty());
    }
    // FIXME: allow an empty hole next to a non-empty exterior.
    // GEOS accepts it, and the OGC spec does not rule it out.
    polygon
        .rings()
        .all(|ring| has_at_least_distinct_vertices(ring.coords, ring.vertices, MIN_RING_VERTICES))
}

/// [polygon_shape_is_valid] for each of `polygons`, made of `rings`.
#[cfg(feature = "geos")]
pub(crate) fn polygon_shapes_are_valid(
    polygons: &ValidList,
    rings: &ValidList,
    coords: &ValidCoords,
) -> Bitmap {
    Bitmap::from_iter(polygons.ranges().map(|parts| {
        polygon_shape_is_valid(&ScalarPolygon {
            rings,
            coords,
            parts,
        })
    }))
}
