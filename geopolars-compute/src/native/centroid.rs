//! The centroid of a linestring or polygon, or their multi-variants, in one pass over the storage.
//! Implements the Python `centroid`, which documents how every kind is weighted.
//! - polygon uses the shoelace formula.
//! - multi-geometry's running sum just goes on over its parts.
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

/// Adds the moment of a closed ring, with the shoelace formula.
/// x and y are taken relative to the first vertex `p0`, so rings far from the origin keep their precision.
/// With cross = x_i y_{i+1} - x_{i+1} y_i over those, a = Σ cross is twice the area,
/// and the centroid is p0 + (Σ (x_i + x_{i+1}) cross, Σ (y_i + y_{i+1}) cross) / 3a.
/// As each cross is twice the area of a triangle fanning out from `p0`,
/// z and m are averaged over those triangles at the mean of their corners: Σ cross (z_0 + z_i + z_{i+1}) / 3a.
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
    let shift = [p0[0], p0[1], 0.0, 0.0];
    let shifted = |p: [f64; 4]| [p[0] - shift[0], p[1] - shift[1], p[2], p[3]];
    let (mut a, mut sum) = (0.0, [0.0; 4]);
    let mut p = shifted(p0);
    // The repeated last vertex is only ever the `q`.
    for q in coordinates {
        let q = shifted(q);
        let cross = p[0] * q[1] - q[0] * p[1];
        a += cross;
        sum[0] += (p[0] + q[0]) * cross;
        sum[1] += (p[1] + q[1]) * cross;
        for k in 2..4 {
            if has_axis::<Z, M>(k) {
                sum[k] += cross * (p0[k] + p[k] + q[k]);
            }
        }
        p = q;
    }
    let sign = sign * a.signum();
    for (k, (total, s)) in moment.sum.iter_mut().zip(sum).enumerate() {
        if has_axis::<Z, M>(k) {
            *total += sign * (s / 6.0 + shift[k] * a / 2.0);
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
