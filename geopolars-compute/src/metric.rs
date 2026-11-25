//! Different methods for measuring distance.
//! Multiple methods are possible:
//! - Planar
//! - Geodesic

// TODO: A couple more methods of calculating distance:
// - Euclidian (fast like planar, but now uses the CRS units),
//   which should have a usage note that it might have a large error for long distances.
// - Haversine, errors up to 0.5%
// - Rhumb, distance along const lat or long bearing
// - Projected, which is just Geodesic but first projected to WGS84, and uses Karney's algorithm.

#[cfg(feature = "native")]
use std::ops::Range;

use geographiclib_rs::{Geodesic, InverseGeodesic};
#[cfg(feature = "native")]
use geographiclib_rs::{PolygonArea, Winding};
#[cfg(feature = "native")]
use geopolars_views::{ScalarLineString, ValidCoords};

#[cfg(feature = "native")]
use crate::native::AreaMetric;
#[cfg(feature = "native")]
use crate::proj::ReprojectedXy;
use crate::proj::{reproject_position, Transformation};
#[cfg(feature = "native")]
use crate::GeoComputeError;
use crate::GeoComputeResult;

/// Marker trait so we can trace all the metrics to one place :)
pub trait BaseMetric {}

/// In the space the coordinates lie in, by Pythagoras.
pub struct Planar;

impl BaseMetric for Planar {}

/// A point's `(x, y)` and its `z`, kept apart so the `(x, y)` can go to PROJ as is.
pub type WithHeight = ((f64, f64), f64);

/// Measures coordinates in one CRS against each other along the ellipsoid that CRS lies on.
pub struct GeodesicMetric {
    to_lonlat: Transformation,
    ellipsoid: Geodesic,
}

impl BaseMetric for GeodesicMetric {}

impl GeodesicMetric {
    /// Measures along the ellipsoid of `semi_major` and `flattening`, in the unit of `semi_major`.
    ///
    /// `to_lonlat` must go from the CRS to its own longitude/latitude, so no datum is shifted.
    /// `semi_major` must be in the unit of the CRS's `x` and `y` (metres if those are degrees).
    pub fn new(to_lonlat: Transformation, semi_major: f64, flattening: f64) -> Self {
        Self {
            to_lonlat,
            ellipsoid: Geodesic::new(semi_major, flattening),
        }
    }

    /// `(lon, lat)` in degrees, on this CRS's own datum.
    ///
    /// Reprojecting is the expensive part of measuring,
    /// so something measuring many segments should do this once per vertex.
    pub fn lonlat(&self, (x, y): (f64, f64)) -> GeoComputeResult<(f64, f64)> {
        // TODO: `length` and `distance` still call this once per vertex.
        // Batch them through `ReprojectedXy`, as `area` does.
        let [lon, lat, _] = reproject_position(&self.to_lonlat, [x, y, 0.0])?;
        Ok((lon, lat))
    }

    /// The geodesic distance between two coordinates already in [`lonlat`](Self::lonlat),
    /// in the unit the ellipsoid was given in (see [`new`](Self::new)).
    pub fn between(&self, (lon_a, lat_a): (f64, f64), (lon_b, lat_b): (f64, f64)) -> f64 {
        self.ellipsoid.inverse(lat_a, lon_a, lat_b, lon_b)
    }
}

#[cfg(feature = "native")]
impl AreaMetric for GeodesicMetric {
    type Error = GeoComputeError;
    type Chunk = ReprojectedXy;
    const CHECKS_COORDS_EARLY: bool = true;

    fn prepare_chunk(&self, coords: &ValidCoords, vertices: Range<usize>) -> ReprojectedXy {
        ReprojectedXy::new(&self.to_lonlat, coords, vertices)
    }

    #[inline(always)]
    fn check_row(
        &self,
        lonlats: &ReprojectedXy,
        coords: &ValidCoords,
        vertices: Range<usize>,
    ) -> Result<(), GeoComputeError> {
        lonlats.check(&self.to_lonlat, coords, vertices)
    }

    fn ring_area(&self, lonlats: &ReprojectedXy, ring: &ScalarLineString) -> f64 {
        ring_signed_area(&self.ellipsoid, lonlats, ring).abs()
    }
}

/// The area `ring` encloses on `ellipsoid`, positive when its smaller side lies to its left.
#[cfg(feature = "native")]
#[inline(always)]
fn ring_signed_area(ellipsoid: &Geodesic, lonlats: &ReprojectedXy, ring: &ScalarLineString) -> f64 {
    // On the ellipsoid, in the CRS's unit squared, like [`between`](GeodesicMetric::between).
    // Based on `geographiclib_rs`.
    let mut area = PolygonArea::new(ellipsoid, Winding::CounterClockwise);
    for i in ring.vertices.clone() {
        let (lon, lat) = lonlats.xy(i);
        // The repeated closing vertex adds an edge of length zero, which encloses nothing.
        area.add_point(lat, lon);
    }
    // Signed, so that it lies within half the ellipsoid.
    // Unsigned would take a clockwise ring to enclose everything outside of it.
    let (_perimeter, area, _vertices) = area.compute(true);
    area
}
