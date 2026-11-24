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
//!
//! ## `ANCHOR`
//!
//! The [CentroidAnchor] that every ring of a polygon fans out from.

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

const_enum!(
    /// Which point every ring of a polygon is fanned out from.
    /// Implements `anchor` of the Python `centroid`.
    pub enum CentroidAnchor {
        LengthWeightedMean,
        VertexMean,
    }
);

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

/// The point a ring fans out from, relative to its first vertex.
struct RingAnchor<const Z: bool, const M: bool, const ANCHOR: u8> {
    sum: [f64; 4],
    weight: f64,
}

impl<const Z: bool, const M: bool, const ANCHOR: u8> RingAnchor<Z, M, ANCHOR> {
    const ZERO: Self = Self {
        sum: [0.0; 4],
        weight: 0.0,
    };

    #[inline(always)]
    fn add_edge(
        &mut self,
        d_p: [f64; 4],
        d_q: [f64; 4],
        segment_length: impl Fn([f64; 4], [f64; 4]) -> f64,
    ) {
        let (weight, centre) = match const { CentroidAnchor::from_u8(ANCHOR) } {
            // On a closed ring, every vertex ends exactly one edge.
            CentroidAnchor::VertexMean => (1.0, d_q),
            CentroidAnchor::LengthWeightedMean => (
                segment_length(d_p, d_q),
                std::array::from_fn(|k| 0.5 * (d_p[k] + d_q[k])),
            ),
        };
        for k in 0..4 {
            if has_axis::<Z, M>(k) {
                self.sum[k] += weight * centre[k];
            }
        }
        self.weight += weight;
    }

    #[inline(always)]
    fn relative_to_first(&self) -> [f64; 4] {
        // A ring without length has no area either, so any point does.
        if self.weight == 0.0 {
            return [0.0; 4];
        }
        self.sum.map(|s| s / self.weight)
    }
}

/// Adds the area of a closed ring, and its area-weighted centre, to `moment`.
/// A hole passes `sign = -1.0`, which takes its area out again.
#[inline(always)]
fn add_ring<const Z: bool, const M: bool, const ANCHOR: u8>(
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
    // The area, x, and y come out the same whichever point the fan starts from.
    // As z and m need not lie on a plane, they do depend on it.
    // `sum_dx` and `sum_dy` let them move to the fan from the anchor after the loop.
    let (mut a, mut sum) = (0.0, [0.0; 4]);
    let mut anchor = RingAnchor::<Z, M, ANCHOR>::ZERO;
    let (mut sum_dx, mut sum_dy) = ([0.0; 4], [0.0; 4]);
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
        if Z || M {
            anchor.add_edge(d_p, d_q, footprint_segment_length);
            for k in 2..4 {
                if has_axis::<Z, M>(k) {
                    sum_dx[k] += (d_q[0] - d_p[0]) * (d_p[k] + d_q[k]);
                    sum_dy[k] += (d_q[1] - d_p[1]) * (d_p[k] + d_q[k]);
                }
            }
        }
        d_p = d_q;
    }
    if Z || M {
        // Moving the tip of a triangle from `p0` to `c` takes `c × (d_q - d_p)` off its `fan`.
        // Over a closed ring, these add up to nothing, so `a` stays the same.
        let c = anchor.relative_to_first();
        for k in 2..4 {
            if has_axis::<Z, M>(k) {
                sum[k] += c[k] * a - (c[0] * sum_dy[k] - c[1] * sum_dx[k]);
            }
        }
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
fn add_polygon<const Z: bool, const M: bool, const ANCHOR: u8>(
    moment: &mut Moment<Z, M>,
    polygon: &ScalarPolygon,
) {
    let Some(exterior) = polygon.exterior() else {
        return;
    };
    add_ring::<Z, M, ANCHOR>(moment, &exterior, 1.0);
    for hole in polygon.holes() {
        add_ring::<Z, M, ANCHOR>(moment, &hole, -1.0);
    }
}

#[inline(always)]
fn xyz(a: [f64; 4]) -> [f64; 3] {
    [a[0], a[1], a[2]]
}

#[inline(always)]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The triangles a ring fans out into in 3D, summed before they are projected onto a plane.
struct RingFanThreeD {
    /// Σ n, twice the vector area of the ring.
    normal: [f64; 3],
    /// Σ n[j] * (the sum of the corners)[k], for `j` in x, y, z and `k` in x, y, z, m.
    moment: [[f64; 4]; 3],
}

/// Fans `ring` out from its anchor in 3D, as a [RingFanThreeD] in absolute coordinates.
#[inline(always)]
fn ring_fan_three_d<const M: bool, const ANCHOR: u8>(ring: &ScalarLineString) -> RingFanThreeD {
    let mut fan = RingFanThreeD {
        normal: [0.0; 3],
        moment: [[0.0; 4]; 3],
    };
    let mut coordinates = ring.coordinates::<true, M>();
    let Some(p0) = coordinates.next() else {
        return fan;
    };
    // As in `add_ring`, the triangles fan out from `p0`, relative to `p0`.
    // The cross product `n` of a triangle is twice its vector area.
    // `slope` lets the fan move to the anchor after the loop.
    let mut anchor = RingAnchor::<true, M, ANCHOR>::ZERO;
    let mut slope = [[0.0; 4]; 3];
    let mut d_p = [0.0; 4];
    for q in coordinates {
        let d_q: [f64; 4] = std::array::from_fn(|k| {
            if has_axis::<true, M>(k) {
                q[k] - p0[k]
            } else {
                0.0
            }
        });
        let n = cross(xyz(d_p), xyz(d_q));
        anchor.add_edge(d_p, d_q, three_d_segment_length);
        for j in 0..3 {
            fan.normal[j] += n[j];
            for k in 0..4 {
                if has_axis::<true, M>(k) {
                    let corners = d_p[k] + d_q[k];
                    fan.moment[j][k] += n[j] * corners;
                    slope[j][k] += (d_q[j] - d_p[j]) * corners;
                }
            }
        }
        d_p = d_q;
    }
    // Moving the tip of a triangle from `p0` to `c` takes `c × (d_q - d_p)` off its `n`.
    // Over a closed ring, these add up to nothing, so `normal` stays the same.
    let c = anchor.relative_to_first();
    for k in 0..4 {
        if has_axis::<true, M>(k) {
            let shift = cross(xyz(c), [slope[0][k], slope[1][k], slope[2][k]]);
            for j in 0..3 {
                fan.moment[j][k] += (c[k] + 3.0 * p0[k]) * fan.normal[j] - shift[j];
            }
        }
    }
    fan
}

/// moment += (exterior - holes), measured on the plane facing along the polygon's vector area.
#[inline(always)]
fn add_polygon_three_d<const M: bool, const ANCHOR: u8>(
    moment: &mut Moment<true, M>,
    polygon: &ScalarPolygon,
) {
    let mut rings = polygon.rings();
    let Some(exterior) = rings.next() else {
        return;
    };
    let mut fan = ring_fan_three_d::<M, ANCHOR>(&exterior);
    let exterior_normal = fan.normal;
    for hole in rings {
        let hole = ring_fan_three_d::<M, ANCHOR>(&hole);
        // The hole is taken out whichever way it winds.
        let sign = dot(hole.normal, exterior_normal).signum();
        for j in 0..3 {
            fan.normal[j] -= sign * hole.normal[j];
            for k in 0..4 {
                fan.moment[j][k] -= sign * hole.moment[j][k];
            }
        }
    }
    // A triangle's area on the plane is `n · normal / (2 |normal|)`, at the mean of its corners.
    let squared = dot(fan.normal, fan.normal);
    if squared == 0.0 {
        return;
    }
    let length = squared.sqrt();
    for k in 0..4 {
        if has_axis::<true, M>(k) {
            let along_normal: f64 = (0..3).map(|j| fan.normal[j] * fan.moment[j][k]).sum();
            moment.sum[k] += along_normal / (6.0 * length);
        }
    }
    moment.weight += length / 2.0;
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
    anchor: CentroidAnchor,
) -> PointArray {
    with_const_flags!(
        ANCHOR: CentroidAnchor = anchor;
        match method {
            CentroidMethod::ThreeD if Z => polygon_centroids_three_d::<M, ANCHOR>(array),
            // Without a z, a polygon's own plane is its footprint.
            CentroidMethod::Footprint | CentroidMethod::ThreeD => {
                polygon_centroids_footprint::<Z, M, ANCHOR>(array)
            }
        }
    )
}

fn polygon_centroids_footprint<const Z: bool, const M: bool, const ANCHOR: u8>(
    array: &Untrimmed<PolygonArray>,
) -> PointArray {
    centroids::<Z, M>(array.rows(), |moment, i| {
        add_polygon::<Z, M, ANCHOR>(moment, &array.get(i));
    })
}

fn polygon_centroids_three_d<const M: bool, const ANCHOR: u8>(
    array: &Untrimmed<PolygonArray>,
) -> PointArray {
    centroids::<true, M>(array.rows(), |moment, i| {
        add_polygon_three_d::<M, ANCHOR>(moment, &array.get(i));
    })
}

pub fn multi_polygon_centroids<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiPolygonArray>,
    method: CentroidMethod,
    anchor: CentroidAnchor,
) -> PointArray {
    with_const_flags!(
        ANCHOR: CentroidAnchor = anchor;
        match method {
            CentroidMethod::ThreeD if Z => multi_polygon_centroids_three_d::<M, ANCHOR>(array),
            // Without a z, a polygon's own plane is its footprint.
            CentroidMethod::Footprint | CentroidMethod::ThreeD => {
                multi_polygon_centroids_footprint::<Z, M, ANCHOR>(array)
            }
        }
    )
}

fn multi_polygon_centroids_footprint<const Z: bool, const M: bool, const ANCHOR: u8>(
    array: &Untrimmed<MultiPolygonArray>,
) -> PointArray {
    centroids::<Z, M>(array.rows(), |moment, i| {
        for polygon in array.get(i).polygons() {
            add_polygon::<Z, M, ANCHOR>(moment, &polygon);
        }
    })
}

fn multi_polygon_centroids_three_d<const M: bool, const ANCHOR: u8>(
    array: &Untrimmed<MultiPolygonArray>,
) -> PointArray {
    centroids::<true, M>(array.rows(), |moment, i| {
        for polygon in array.get(i).polygons() {
            add_polygon_three_d::<M, ANCHOR>(moment, &polygon);
        }
    })
}
