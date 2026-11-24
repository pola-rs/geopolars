//! The centroid of a linestring or polygon, or their multi-variants, in one pass over the storage.
//! Implements the Python `centroid`, which documents how every kind is weighted.
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! If `Z=true`, the z values are averaged too.
//!
//! ## `M`
//!
//! If `M=true`, the m values are averaged too.

use std::convert::Infallible;

use geopolars_views::{
    Coords, LineStringArray, List, MultiLineStringArray, MultiPolygonArray, PointArray,
    PolygonArray, ScalarLineString, ScalarPolygon, Untrimmed, ValidCoords,
};
use polars_arrow::bitmap::Bitmap;

use crate::per_row::per_row_values;

/// Which axes measure the length or area that a centroid is weighted by.
/// Implements `method` of the Python `centroid`.
#[derive(Clone, Copy, Debug)]
pub enum CentroidMethod {
    Footprint,
    ThreeD,
}

/// Used at compile to drop loops
/// Whether axis `k` is there.
/// k=0 is x, k=1 is y etc
#[inline(always)]
const fn has_axis<const Z: bool, const M: bool>(k: usize) -> bool {
    k < 2 || (k == 2 && Z) || (k == 3 && M)
}

/// Running average for a weight.
/// You can keep adding to this!
struct Moment<const Z: bool, const M: bool> {
    /// Σ weight * centre, in x, y, z, m.
    /// An axis that is not [there](has_axis) stays `0.0`.
    sum: [f64; 4],
    /// Σ weight
    /// All share the same
    weight: f64,
}

impl<const Z: bool, const M: bool> Moment<Z, M> {
    const ZERO: Self = Self {
        sum: [0.0; 4],
        weight: 0.0,
    };

    /// Σ weight * centre / Σ weight.
    /// `None` if there is no length or area, but a NaN or infinite weight propagates.
    #[inline(always)]
    fn average(&self) -> Option<[f64; 4]> {
        // A hole with an infinite coordinate can weigh -inf.
        let no_weight = self.weight <= 0.0 && self.weight.is_finite();
        (!no_weight).then(|| {
            std::array::from_fn(|k| {
                if has_axis::<Z, M>(k) {
                    self.sum[k] / self.weight
                } else {
                    0.0
                }
            })
        })
    }
}

#[inline(always)]
fn add_line<const Z: bool, const M: bool>(
    moment: &mut Moment<Z, M>,
    line: &ScalarLineString,
    segment_length: impl Fn([f64; 4], [f64; 4]) -> f64,
) {
    let mut coordinates = line.coordinates::<Z, M>();
    let Some(mut previous) = coordinates.next() else {
        return;
    };
    for here in coordinates {
        let len = segment_length(previous, here);
        for k in 0..4 {
            if has_axis::<Z, M>(k) {
                moment.sum[k] += 0.5 * len * (previous[k] + here[k]);
            }
        }
        moment.weight += len;
        previous = here;
    }
}

#[inline(always)]
fn footprint_segment_length(a: [f64; 4], b: [f64; 4]) -> f64 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

#[inline(always)]
fn three_d_segment_length(a: [f64; 4], b: [f64; 4]) -> f64 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
}

/// Adds the area of a closed ring, and its area-weighted centre, to `moment`.
/// A hole passes `sign = -1.0`, which takes its area out again.
#[inline(always)]
fn add_ring<const Z: bool, const M: bool>(
    moment: &mut Moment<Z, M>,
    ring: &ScalarLineString,
    sign: f64,
) {
    let mut coordinates = ring.coordinates::<Z, M>();
    let Some(p0) = coordinates.next() else {
        return;
    };
    // The ring is split into triangles that fan out from `p0`.
    // Each weighs half its cross product `fan`, at the mean of its corners.
    // Taking the coordinates relative to `p0` keeps their precision far from the origin.
    // `p0` itself is added back once, after the loop.
    let (mut a, mut sum) = (0.0, [0.0; 4]);
    let mut d_p = [0.0; 4];
    // The repeated last vertex is only ever the `q`.
    for q in coordinates {
        let d_q: [f64; 4] = std::array::from_fn(|k| {
            if has_axis::<Z, M>(k) {
                q[k] - p0[k]
            } else {
                0.0
            }
        });
        let fan = d_p[0] * d_q[1] - d_q[0] * d_p[1];
        a += fan;
        for k in 0..4 {
            if has_axis::<Z, M>(k) {
                sum[k] += fan * (d_p[k] + d_q[k]);
            }
        }
        d_p = d_q;
    }
    let sign = sign * a.signum();
    for (k, (total, s)) in moment.sum.iter_mut().zip(sum).enumerate() {
        if has_axis::<Z, M>(k) {
            *total += sign * (s / 6.0 + p0[k] * a / 2.0);
        }
    }
    moment.weight += sign * a / 2.0;
}

/// moment += (exterior - holes)
#[inline(always)]
fn add_polygon<const Z: bool, const M: bool>(moment: &mut Moment<Z, M>, polygon: &ScalarPolygon) {
    let Some(exterior) = polygon.exterior() else {
        return;
    };
    add_ring(moment, &exterior, 1.0);
    for hole in polygon.holes() {
        add_ring(moment, &hole, -1.0);
    }
}

/// The centroid of every row of `rows`, of the moment `add_row` adds for it.
#[inline(always)]
fn centroids<const Z: bool, const M: bool>(
    rows: &List,
    add_row: impl Fn(&mut Moment<Z, M>, usize),
) -> PointArray {
    let Ok(centroids) = per_row_values(rows, |i| {
        let mut moment = Moment::ZERO;
        add_row(&mut moment, i);
        Ok::<_, Infallible>(moment.average())
    });
    let axis = |k: usize| -> Vec<f64> {
        centroids
            .iter()
            .map(|centroid| centroid.map_or(0.0, |centroid| centroid[k]))
            .collect()
    };
    let axes = (axis(0), axis(1), Z.then(|| axis(2)), M.then(|| axis(3)));
    let validity = Bitmap::from_iter(centroids.iter().map(Option::is_some));
    PointArray::new(Coords::from_valid(
        ValidCoords::from_axes(axes),
        Some(validity),
    ))
}

pub fn line_string_centroids<const Z: bool, const M: bool>(
    array: &Untrimmed<LineStringArray>,
    method: CentroidMethod,
) -> PointArray {
    match method {
        CentroidMethod::ThreeD if Z => line_string_centroids_three_d::<M>(array),
        // Without a z, measuring in 3D is measuring the footprint.
        CentroidMethod::Footprint | CentroidMethod::ThreeD => {
            line_string_centroids_footprint::<Z, M>(array)
        }
    }
}

fn line_string_centroids_footprint<const Z: bool, const M: bool>(
    array: &Untrimmed<LineStringArray>,
) -> PointArray {
    centroids::<Z, M>(array.rows(), |moment, i| {
        add_line(moment, &array.get(i), footprint_segment_length);
    })
}

fn line_string_centroids_three_d<const M: bool>(array: &Untrimmed<LineStringArray>) -> PointArray {
    centroids::<true, M>(array.rows(), |moment, i| {
        add_line(moment, &array.get(i), three_d_segment_length);
    })
}

pub fn multi_line_string_centroids<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiLineStringArray>,
    method: CentroidMethod,
) -> PointArray {
    match method {
        CentroidMethod::ThreeD if Z => multi_line_string_centroids_three_d::<M>(array),
        // Without a z, measuring in 3D is measuring the footprint.
        CentroidMethod::Footprint | CentroidMethod::ThreeD => {
            multi_line_string_centroids_footprint::<Z, M>(array)
        }
    }
}

fn multi_line_string_centroids_footprint<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiLineStringArray>,
) -> PointArray {
    centroids::<Z, M>(array.rows(), |moment, i| {
        for line in array.get(i).line_strings() {
            add_line(moment, &line, footprint_segment_length);
        }
    })
}

fn multi_line_string_centroids_three_d<const M: bool>(
    array: &Untrimmed<MultiLineStringArray>,
) -> PointArray {
    centroids::<true, M>(array.rows(), |moment, i| {
        for line in array.get(i).line_strings() {
            add_line(moment, &line, three_d_segment_length);
        }
    })
}

pub fn polygon_centroids<const Z: bool, const M: bool>(
    array: &Untrimmed<PolygonArray>,
    method: CentroidMethod,
) -> PointArray {
    match method {
        CentroidMethod::ThreeD if Z => polygon_centroids_three_d::<M>(array),
        // Without a z, a polygon's own plane is its footprint.
        CentroidMethod::Footprint | CentroidMethod::ThreeD => {
            polygon_centroids_footprint::<Z, M>(array)
        }
    }
}

fn polygon_centroids_footprint<const Z: bool, const M: bool>(
    array: &Untrimmed<PolygonArray>,
) -> PointArray {
    centroids::<Z, M>(array.rows(), |moment, i| add_polygon(moment, &array.get(i)))
}

fn polygon_centroids_three_d<const M: bool>(_array: &Untrimmed<PolygonArray>) -> PointArray {
    todo!("weigh a polygon by its area on its own plane")
}

pub fn multi_polygon_centroids<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiPolygonArray>,
    method: CentroidMethod,
) -> PointArray {
    match method {
        CentroidMethod::ThreeD if Z => multi_polygon_centroids_three_d::<M>(array),
        // Without a z, a polygon's own plane is its footprint.
        CentroidMethod::Footprint | CentroidMethod::ThreeD => {
            multi_polygon_centroids_footprint::<Z, M>(array)
        }
    }
}

fn multi_polygon_centroids_footprint<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiPolygonArray>,
) -> PointArray {
    centroids::<Z, M>(array.rows(), |moment, i| {
        for polygon in array.get(i).polygons() {
            add_polygon(moment, &polygon);
        }
    })
}

fn multi_polygon_centroids_three_d<const M: bool>(
    _array: &Untrimmed<MultiPolygonArray>,
) -> PointArray {
    todo!("weigh every polygon by its area on its own plane")
}
