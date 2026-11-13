//! The centroid of a linestring or polygon, or their multi-variants, in one pass over the storage.
//! Implements the Python `centroid`, which documents how every kind is weighted.
//! A polygon uses the shoelace formula.
//! A multi-geometry's running sum just goes on over its parts.
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! Every function here is generic over `Z`.
//! If `Z=true`, the z values are averaged too.
//! A linestring's segments are then measured in x, y and z.
//!
//! ## `M`
//!
//! Every function here is generic over `M`.
//! If `M=true`, the m values are averaged too, with the same weights as the other axes.

use std::convert::Infallible;

use geopolars_views::{
    Coords, LineStringArray, List, MultiLineStringArray, MultiPolygonArray, PointArray,
    PolygonArray, ScalarLineString, ScalarPolygon, Untrimmed, ValidCoords,
};
use polars_arrow::bitmap::Bitmap;

use crate::per_row::per_row_values;

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
    #[inline(always)]
    fn average(&self) -> Option<[f64; 4]> {
        (self.weight > 0.0).then(|| {
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

/// Adds `line` to the moment
/// Σ len (a + b) / 2 and Σ len over its segments `ab`.
#[inline(always)]
fn add_line<const Z: bool, const M: bool>(moment: &mut Moment<Z, M>, line: &ScalarLineString) {
    let mut coordinates = line.coordinates::<Z, M>();
    let Some(mut previous) = coordinates.next() else {
        return;
    };
    for here in coordinates {
        let mut squared = (here[0] - previous[0]).powi(2) + (here[1] - previous[1]).powi(2);
        if Z {
            squared += (here[2] - previous[2]).powi(2);
        }
        let len = squared.sqrt();
        for k in 0..4 {
            if has_axis::<Z, M>(k) {
                moment.sum[k] += 0.5 * len * (previous[k] + here[k]);
            }
        }
        moment.weight += len;
        previous = here;
    }
}

/// Moment += that of a (closed) ring.
/// shoelace formula:
/// Over the ring, a = Σ cross, mx = Σ (x_i + x_{i+1}) cross and my = Σ (y_i + y_{i+1}) cross,
/// with cross = x_i y_{i+1} - x_{i+1} y_i: its centroid is (mx, my) / 3a,
/// so weighted by its area a / 2, (mx, my) / 6.
/// z and m are averaged over the area too:
/// over the triangles fanning out from the first vertex `p0`,
/// each weighted by its area, fan / 2 (with fan the cross of `p_i - p0` and `p_{i+1} - p0`),
/// at the mean of its corners, (z_0 + z_i + z_{i+1}) / 3: Σ fan (z_0 + z_i + z_{i+1}) / 6.
/// On a flat ring, that is the height of its plane above the centroid.
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
    let (mut a, mut sum) = (0.0, [0.0; 4]);
    let mut p = p0;
    // The repeated last vertex is only ever the `q`.
    for q in coordinates {
        let cross = p[0] * q[1] - q[0] * p[1];
        a += cross;
        sum[0] += (p[0] + q[0]) * cross;
        sum[1] += (p[1] + q[1]) * cross;
        if Z || M {
            let fan = (p[0] - p0[0]) * (q[1] - p0[1]) - (q[0] - p0[0]) * (p[1] - p0[1]);
            for k in 2..4 {
                if has_axis::<Z, M>(k) {
                    sum[k] += fan * (p0[k] + p[k] + q[k]);
                }
            }
        }
        p = q;
    }
    let sign = sign * a.signum();
    for (k, (total, s)) in moment.sum.iter_mut().zip(sum).enumerate() {
        if has_axis::<Z, M>(k) {
            *total += sign * s / 6.0;
        }
    }
    moment.weight += sign * a / 2.0;
}

/// moment += (exterior - holes)
#[inline(always)]
fn add_polygon<const Z: bool, const M: bool>(moment: &mut Moment<Z, M>, polygon: &ScalarPolygon) {
    let mut rings = polygon.rings();
    let Some(exterior) = rings.next() else {
        return;
    };
    add_ring(moment, &exterior, 1.0);
    for hole in rings {
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
) -> PointArray {
    centroids::<Z, M>(array.rows(), |moment, i| add_line(moment, &array.get(i)))
}

pub fn multi_line_string_centroids<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiLineStringArray>,
) -> PointArray {
    centroids::<Z, M>(array.rows(), |moment, i| {
        for line in array.get(i).line_strings() {
            add_line(moment, &line);
        }
    })
}

pub fn polygon_centroids<const Z: bool, const M: bool>(
    array: &Untrimmed<PolygonArray>,
) -> PointArray {
    centroids::<Z, M>(array.rows(), |moment, i| add_polygon(moment, &array.get(i)))
}

pub fn multi_polygon_centroids<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiPolygonArray>,
) -> PointArray {
    centroids::<Z, M>(array.rows(), |moment, i| {
        for polygon in array.get(i).polygons() {
            add_polygon(moment, &polygon);
        }
    })
}
