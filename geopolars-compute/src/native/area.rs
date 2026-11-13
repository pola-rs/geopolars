//! The area a polygon encloses, as an [AreaMetric] measures it.
//! Implements the Python `area_planar` and `area_geodesic`.

use std::convert::Infallible;

use geographiclib_rs::{PolygonArea, Winding};
use geopolars_views::{
    MultiPolygonArray, PolygonArray, ScalarLineString, ScalarPolygon, Untrimmed,
};
use polars_arrow::array::PrimitiveArray;

use crate::per_row::per_row;
use crate::{BaseMetric, GeoComputeError, GeodesicMetric, Planar};

/// How the area a ring encloses is measured
pub trait AreaMetric: BaseMetric {
    /// Why a ring could not be measured (e.g. a failed reprojection).
    type Error;
    /// The area the closed `ring` encloses, whichever way it winds. Its heights do not count.
    fn ring_area(&self, ring: &ScalarLineString) -> Result<f64, Self::Error>;
}

impl AreaMetric for Planar {
    type Error = Infallible;

    fn ring_area(&self, ring: &ScalarLineString) -> Result<f64, Infallible> {
        Ok(twice_signed_area(ring).abs() / 2.0)
    }
}

/// Twice the area the closed `ring` encloses in x and y, positive when it runs counter-clockwise.
#[inline(always)]
pub(super) fn twice_signed_area(ring: &ScalarLineString) -> f64 {
    // By the shoelace formula, relative to the ring's first vertex,
    // so coordinates far from the origin keep their precision.
    // The repeated closing vertex adds nothing.
    let mut positions = ring.positions::<false>();
    let Some([x0, y0, _]) = positions.next() else {
        return 0.0;
    };
    let (mut x_previous, mut y_previous) = (0.0, 0.0);
    let mut twice_signed = 0.0;
    for [x, y, _] in positions {
        let (x, y) = (x - x0, y - y0);
        twice_signed += x_previous * y - x * y_previous;
        (x_previous, y_previous) = (x, y);
    }
    twice_signed
}

impl AreaMetric for GeodesicMetric {
    type Error = GeoComputeError;

    fn ring_area(&self, ring: &ScalarLineString) -> Result<f64, GeoComputeError> {
        // On the ellipsoid, in the CRS's unit squared, like [`between`](GeodesicMetric::between).
        // Based on `geographiclib_rs`.
        let mut area = PolygonArea::new(&self.ellipsoid, Winding::CounterClockwise);
        for [x, y, _] in ring.positions::<false>() {
            let (lon, lat) = self.lonlat((x, y))?;
            // The repeated closing vertex adds an edge of length zero, which encloses nothing.
            area.add_point(lat, lon);
        }
        // Signed, then made positive.
        // unsigned would take a clockwise ring
        // to enclose everything outside of it.
        let (_perimeter, area, _vertices) = area.compute(true);
        Ok(area.abs())
    }
}

#[inline(always)]
fn polygon_area<M: AreaMetric>(metric: &M, polygon: &ScalarPolygon) -> Result<f64, M::Error> {
    let mut rings = polygon.rings();
    let Some(exterior) = rings.next() else {
        return Ok(0.0);
    };
    let exterior = metric.ring_area(&exterior)?;
    let holes = rings
        .map(|hole| metric.ring_area(&hole))
        .sum::<Result<f64, _>>()?;
    Ok(exterior - holes)
}

/// The area of every polygon of a chunk.
pub fn polygon_areas<M: AreaMetric>(
    metric: &M,
    array: &Untrimmed<PolygonArray>,
) -> Result<PrimitiveArray<f64>, M::Error> {
    per_row(array.rows(), |i| polygon_area(metric, &array.get(i)))
}

/// The area of every multipolygon of a chunk.
pub fn multi_polygon_areas<M: AreaMetric>(
    metric: &M,
    array: &Untrimmed<MultiPolygonArray>,
) -> Result<PrimitiveArray<f64>, M::Error> {
    per_row(array.rows(), |i| {
        array
            .get(i)
            .polygons()
            .map(|polygon| polygon_area(metric, &polygon))
            .sum()
    })
}
