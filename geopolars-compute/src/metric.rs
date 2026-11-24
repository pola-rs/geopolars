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

use std::ops::Range;

use geographiclib_rs::{Geodesic, InverseGeodesic};
use geopolars_views::ValidCoords;

use crate::proj::{reproject_position, reprojected_xy_with_failures_infinite, Transformation};
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
    /// To the CRS's own longitude/latitude, so no datum is shifted.
    to_lonlat: Transformation,
    pub(crate) ellipsoid: Geodesic,
}

impl BaseMetric for GeodesicMetric {}

impl GeodesicMetric {
    /// `to_lonlat` goes from the CRS to its own longitude/latitude, so no datum is shifted.
    /// `semi_major` is in the unit of the CRS's `x` and `y` (metres if those are degrees),
    /// which is the unit everything measured comes out in.
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
        let [lon, lat, _] = reproject_position(&self.to_lonlat, [x, y, 0.0])?;
        Ok((lon, lat))
    }

    /// [`lonlat`](Self::lonlat) for `vertices` of `coords`, reprojected in a single call to PROJ.
    /// Read them back with [`lonlat_of`](Self::lonlat_of).
    pub fn lonlats(&self, coords: &ValidCoords, vertices: Range<usize>) -> LonLats {
        let first_vertex = vertices.start;
        let [lon, lat] = reprojected_xy_with_failures_infinite(&self.to_lonlat, coords, vertices);
        LonLats {
            lon,
            lat,
            first_vertex,
        }
    }

    /// Looks up the [`lonlat`](Self::lonlat) of coordinate `i` in `lonlats`,
    /// reprojecting it on its own where the batch failed.
    ///
    /// `lonlats` must hold coordinate `i` of `coords`.
    #[inline(always)]
    pub fn lonlat_of(
        &self,
        lonlats: &LonLats,
        coords: &ValidCoords,
        i: usize,
    ) -> GeoComputeResult<(f64, f64)> {
        let j = i - lonlats.first_vertex;
        let (lon, lat) = (lonlats.lon[j], lonlats.lat[j]);
        if lon == f64::INFINITY {
            // PROJ failed on it, or it came out infinite without failing.
            // On its own, it gives the error, or the same infinite position again.
            return self.lonlat((coords.x()[i], coords.y()[i]));
        }
        Ok((lon, lat))
    }

    /// The geodesic distance between two coordinates already in [`lonlat`](Self::lonlat),
    /// in the unit the ellipsoid was given in (see [`new`](Self::new)).
    pub fn between(&self, (lon_a, lat_a): (f64, f64), (lon_b, lat_b): (f64, f64)) -> f64 {
        self.ellipsoid.inverse(lat_a, lon_a, lat_b, lon_b)
    }
}

/// The longitudes and latitudes of a run of vertices, made by [`GeodesicMetric::lonlats`].
pub struct LonLats {
    lon: Vec<f64>,
    lat: Vec<f64>,
    /// The index of the first vertex in the coordinates they were made from.
    first_vertex: usize,
}
