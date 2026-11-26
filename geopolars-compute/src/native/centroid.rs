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

use std::ops::Range;

use geopolars_views::{
    Coords, LineStringArray, List, MultiLineStringArray, MultiPolygonArray, PointArray,
    PolygonArray, ScalarLineString, ScalarPolygon, Untrimmed, ValidCoords,
};

use super::DistanceMetric;
use crate::per_row::for_each_row;
use crate::utils::validity_with_null_at;
use crate::{row_or_null, CentroidWeight, GeoComputeError, GeoComputeResult, OnRowError, Planar};

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

    /// Whether the weight is at least `smallest_weight`, or is NaN or infinite, so that it propagates.
    #[inline(always)]
    fn outweighs(&self, smallest_weight: f64) -> bool {
        // A hole with an infinite coordinate can weigh -inf.
        !(self.weight < smallest_weight && self.weight.is_finite())
    }

    /// Σ weight * centre / Σ weight.
    #[inline(always)]
    fn average(&self) -> [f64; 4] {
        std::array::from_fn(|k| {
            if has_axis::<Z, M>(k) {
                self.sum[k] / self.weight
            } else {
                0.0
            }
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
fn planar_segment_length<const Z: bool>(a: [f64; 4], b: [f64; 4]) -> f64 {
    Planar.segment::<Z>(xyz(a), xyz(b))
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
            anchor.add_edge(d_p, d_q, planar_segment_length::<false>);
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
fn ring_fan_three_d<const Z: bool, const M: bool, const ANCHOR: u8>(
    ring: &ScalarLineString,
) -> RingFanThreeD {
    let mut fan = RingFanThreeD {
        normal: [0.0; 3],
        moment: [[0.0; 4]; 3],
    };
    let mut coordinates = ring.coordinates::<Z, M>();
    let Some(p0) = coordinates.next() else {
        return fan;
    };
    // As in `add_ring`, the triangles fan out from `p0`, relative to `p0`.
    // The cross product `n` of a triangle is twice its vector area.
    // `slope` lets the fan move to the anchor after the loop.
    let mut anchor = RingAnchor::<Z, M, ANCHOR>::ZERO;
    let mut slope = [[0.0; 4]; 3];
    let mut d_p = [0.0; 4];
    for q in coordinates {
        let d_q: [f64; 4] = std::array::from_fn(|k| {
            if has_axis::<Z, M>(k) {
                q[k] - p0[k]
            } else {
                0.0
            }
        });
        let n = cross(xyz(d_p), xyz(d_q));
        anchor.add_edge(d_p, d_q, planar_segment_length::<Z>);
        for j in 0..3 {
            fan.normal[j] += n[j];
            for k in 0..4 {
                if has_axis::<Z, M>(k) {
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
        if has_axis::<Z, M>(k) {
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
fn add_polygon_three_d<const Z: bool, const M: bool, const ANCHOR: u8>(
    moment: &mut Moment<Z, M>,
    polygon: &ScalarPolygon,
) {
    let mut rings = polygon.rings();
    let Some(exterior) = rings.next() else {
        return;
    };
    let mut fan = ring_fan_three_d::<Z, M, ANCHOR>(&exterior);
    let exterior_normal = fan.normal;
    for hole in rings {
        let hole = ring_fan_three_d::<Z, M, ANCHOR>(&hole);
        // The hole is taken out whichever way it winds.
        // Against an exterior without area, it is taken out as it winds.
        let sign = if dot(hole.normal, exterior_normal) < 0.0 {
            -1.0
        } else {
            1.0
        };
        for j in 0..3 {
            fan.normal[j] -= sign * hole.normal[j];
            for k in 0..4 {
                fan.moment[j][k] -= sign * hole.moment[j][k];
            }
        }
    }
    // A triangle's area on the plane is `n · normal / (2 |normal|)`, at the mean of its corners.
    let length = norm(fan.normal);
    if length == 0.0 {
        return;
    }
    // Dividing by `length` first keeps the sums from underflowing as soon as the weight does.
    let unit = fan.normal.map(|n| n / length);
    // Holes that take out more than the exterior flip the normal.
    // An exterior without area faces nowhere, so any hole with area takes out more than it holds.
    // Their area then counts against the weight, as in `add_ring`.
    let sign = if dot(unit, exterior_normal) > 0.0 {
        1.0
    } else {
        -1.0
    };
    for k in 0..4 {
        if has_axis::<Z, M>(k) {
            let along_normal: f64 = (0..3).map(|j| unit[j] * fan.moment[j][k]).sum();
            moment.sum[k] += sign * along_normal / 6.0;
        }
    }
    moment.weight += sign * length / 2.0;
}

/// moment += (exterior - holes), measured in 3D if `MEASURE_Z`, or else on its footprint.
#[inline(always)]
fn add_polygon_measured<const Z: bool, const M: bool, const MEASURE_Z: bool, const ANCHOR: u8>(
    moment: &mut Moment<Z, M>,
    polygon: &ScalarPolygon,
) {
    if MEASURE_Z {
        add_polygon_three_d::<Z, M, ANCHOR>(moment, polygon);
    } else {
        add_polygon::<Z, M, ANCHOR>(moment, polygon);
    }
}

/// Whether `method` measures the length or area of `coords` in z too.
/// Without a z, measuring in 3D is measuring the footprint.
fn measures_z(method: CentroidMethod, coords: &ValidCoords) -> bool {
    matches!(method, CentroidMethod::ThreeD) && coords.dim().has_z()
}

/// The length of `v`, scaled first so that its squares do not underflow before its cross products do.
#[inline(always)]
fn norm(v: [f64; 3]) -> f64 {
    let largest = v.iter().fold(0.0_f64, |largest, c| largest.max(c.abs()));
    if largest == 0.0 || !largest.is_finite() {
        return dot(v, v).sqrt();
    }
    let scaled = v.map(|c| c / largest);
    largest * dot(scaled, scaled).sqrt()
}

/// Why a row has no centroid.
#[derive(Clone, Copy)]
enum NoCentroid {
    WithoutWeight,
    Underflow { extent: f64 },
    HolesExceedExterior,
}

/// What a row whose weight is too small to trust turns out to be.
enum LightRow {
    Empty,
    /// A thin row, whose weight is small but whose sums are fine.
    Sliver,
    WithoutCentroid(NoCentroid),
}

/// What the centroids of a column are weighted by,
/// to take a closer look at a row whose weight is too small to trust.
#[derive(Clone, Copy)]
struct Weighing<'a> {
    coords: &'a ValidCoords,
    weighed_by: CentroidWeight,
    measures_z: bool,
    /// Below this, the products the length or area and its sums are made of go subnormal.
    smallest_extent: f64,
    /// A row lighter than this may lie within `smallest_extent` of its first vertex.
    smallest_weight: f64,
}

impl<'a> Weighing<'a> {
    fn new(coords: &'a ValidCoords, weighed_by: CentroidWeight, measures_z: bool) -> Self {
        let (smallest_extent, smallest_weight) = match weighed_by {
            // A length squares the sides of a segment, and its sums multiply two sides.
            CentroidWeight::Length => {
                let extent = f64::MIN_POSITIVE.sqrt();
                (extent, extent)
            }
            // An area's sums multiply three sides.
            // A polygon within `extent` of its first vertex fits in a square of side `2 * extent`.
            CentroidWeight::Area => {
                let extent = f64::MIN_POSITIVE.cbrt();
                (extent, 4.0 * extent * extent)
            }
        };
        Self {
            coords,
            weighed_by,
            measures_z,
            smallest_extent,
            smallest_weight,
        }
    }

    /// Overwrites the centroid of every suspect (a row and its weight, too small to trust)
    /// that has none: with the empty point for a row without vertices.
    /// Any other row without a centroid goes to `on_row_error`, so it is returned or added to `failed`.
    #[cold]
    #[inline(never)]
    fn fix_up_suspects<const Z: bool>(
        self,
        axes: &mut [Vec<f64>; 4],
        suspects: &[(usize, f64)],
        row_vertices: impl Fn(usize) -> Range<usize>,
        on_row_error: OnRowError,
        failed: &mut Vec<usize>,
    ) -> GeoComputeResult<()> {
        for &(i, weight) in suspects {
            let fill = match self.look_at_light_row(row_vertices(i), weight) {
                LightRow::Sliver => continue,
                LightRow::Empty => f64::NAN,
                LightRow::WithoutCentroid(why) => {
                    row_or_null::<(), _>(Err(self.error::<Z>(why)), on_row_error)?;
                    failed.push(i);
                    0.0
                }
            };
            // An axis that is not there is an empty `Vec`.
            for axis in axes.iter_mut() {
                if let Some(value) = axis.get_mut(i) {
                    *value = fill;
                }
            }
        }
        Ok(())
    }

    fn look_at_light_row(self, vertices: Range<usize>, weight: f64) -> LightRow {
        if vertices.is_empty() {
            return LightRow::Empty;
        }
        if weight < 0.0 {
            return LightRow::WithoutCentroid(NoCentroid::HolesExceedExterior);
        }
        let extent = self.extent_up_to_smallest(vertices);
        if extent > 0.0 && extent < self.smallest_extent {
            return LightRow::WithoutCentroid(NoCentroid::Underflow { extent });
        }
        if weight > 0.0 {
            return LightRow::Sliver;
        }
        LightRow::WithoutCentroid(NoCentroid::WithoutWeight)
    }

    /// How far `vertices` lie from the first along a measured axis, at most.
    /// It stops at the first vertex at least `smallest_extent` away, as that rules out an underflow.
    fn extent_up_to_smallest(self, vertices: Range<usize>) -> f64 {
        let z = self.measures_z.then(|| self.coords.z()).flatten();
        let mut extent = 0.0_f64;
        for axis in [Some(self.coords.x()), Some(self.coords.y()), z]
            .into_iter()
            .flatten()
        {
            let values = &axis[vertices.clone()];
            let Some(&first) = values.first() else {
                continue;
            };
            for &value in values {
                extent = extent.max((value - first).abs());
                if extent >= self.smallest_extent {
                    return extent;
                }
            }
        }
        extent
    }

    #[cold]
    fn error<const Z: bool>(self, why: NoCentroid) -> GeoComputeError {
        let weighed_by = self.weighed_by;
        match why {
            NoCentroid::WithoutWeight => GeoComputeError::CentroidWithoutWeight {
                weighed_by,
                suggest_3d: Z && !self.measures_z,
            },
            NoCentroid::Underflow { extent } => {
                GeoComputeError::CentroidUnderflow { weighed_by, extent }
            }
            NoCentroid::HolesExceedExterior => GeoComputeError::HolesExceedExterior,
        }
    }
}

/// The number of rows averaged before the suspects among them are looked at again.
const BLOCK_LEN: usize = 1024;

/// The centroid of every row of `rows`, of the moment `add_row` adds for it.
/// A row without a centroid goes to `on_row_error` (see [Weighing::fix_up_suspects]).
#[inline(always)]
fn centroids<const Z: bool, const M: bool>(
    rows: &List,
    row_vertices: impl Fn(usize) -> Range<usize>,
    weighing: Weighing,
    on_row_error: OnRowError,
    add_row: impl Fn(&mut Moment<Z, M>, usize),
) -> GeoComputeResult<PointArray> {
    let len = rows.len();
    let mut axes: [Vec<f64>; 4] =
        std::array::from_fn(|k| Vec::with_capacity(if has_axis::<Z, M>(k) { len } else { 0 }));
    // The rows whose weight is too small to trust, so that the loop itself does no more than average.
    // They are looked at again after each block of rows, while their vertices are still in cache.
    let mut suspects = Vec::new();
    let mut failed = Vec::new();
    let smallest_weight = weighing.smallest_weight;
    for block in (0..len).step_by(BLOCK_LEN) {
        for_each_row(rows, block..len.min(block + BLOCK_LEN), |i, is_valid| {
            // A null row's offsets point anywhere, so it is not measured.
            let average = if is_valid {
                let mut moment = Moment::ZERO;
                add_row(&mut moment, i);
                if !moment.outweighs(smallest_weight) {
                    suspects.push((i, moment.weight));
                }
                moment.average()
            } else {
                [0.0; 4]
            };
            for k in 0..4 {
                if has_axis::<Z, M>(k) {
                    axes[k].push(average[k]);
                }
            }
        });
        if !suspects.is_empty() {
            weighing.fix_up_suspects::<Z>(
                &mut axes,
                &suspects,
                &row_vertices,
                on_row_error,
                &mut failed,
            )?;
            suspects.clear();
        }
    }
    let validity = validity_with_null_at(rows, &failed);
    let [x, y, z, m] = axes;
    Ok(PointArray::new(Coords::from_valid(
        ValidCoords::from_axes((x, y, Z.then_some(z), M.then_some(m))),
        validity,
    )))
}

pub fn line_string_centroids(
    array: &Untrimmed<LineStringArray>,
    method: CentroidMethod,
    on_row_error: OnRowError,
) -> GeoComputeResult<PointArray> {
    let coords = array.coords();
    let dim = coords.dim();
    with_const_flags!(
        Z = dim.has_z(), M = dim.has_m(), MEASURE_Z = measures_z(method, coords);
        line_string_centroids_with::<Z, M, MEASURE_Z>(array, on_row_error)
    )
}

fn line_string_centroids_with<const Z: bool, const M: bool, const MEASURE_Z: bool>(
    array: &Untrimmed<LineStringArray>,
    on_row_error: OnRowError,
) -> GeoComputeResult<PointArray> {
    centroids::<Z, M>(
        array.rows(),
        |i| array.row_vertices(i),
        Weighing::new(array.coords(), CentroidWeight::Length, MEASURE_Z),
        on_row_error,
        |moment, i| add_line(moment, &array.get(i), planar_segment_length::<MEASURE_Z>),
    )
}

pub fn multi_line_string_centroids(
    array: &Untrimmed<MultiLineStringArray>,
    method: CentroidMethod,
    on_row_error: OnRowError,
) -> GeoComputeResult<PointArray> {
    let coords = array.coords();
    let dim = coords.dim();
    with_const_flags!(
        Z = dim.has_z(), M = dim.has_m(), MEASURE_Z = measures_z(method, coords);
        multi_line_string_centroids_with::<Z, M, MEASURE_Z>(array, on_row_error)
    )
}

fn multi_line_string_centroids_with<const Z: bool, const M: bool, const MEASURE_Z: bool>(
    array: &Untrimmed<MultiLineStringArray>,
    on_row_error: OnRowError,
) -> GeoComputeResult<PointArray> {
    centroids::<Z, M>(
        array.rows(),
        |i| array.row_vertices(i),
        Weighing::new(array.coords(), CentroidWeight::Length, MEASURE_Z),
        on_row_error,
        |moment, i| {
            for line in array.get(i).line_strings() {
                add_line(moment, &line, planar_segment_length::<MEASURE_Z>);
            }
        },
    )
}

pub fn polygon_centroids(
    array: &Untrimmed<PolygonArray>,
    method: CentroidMethod,
    anchor: CentroidAnchor,
    on_row_error: OnRowError,
) -> GeoComputeResult<PointArray> {
    let coords = array.coords();
    let dim = coords.dim();
    with_const_flags!(
        Z = dim.has_z(),
        M = dim.has_m(),
        MEASURE_Z = measures_z(method, coords),
        ANCHOR: CentroidAnchor = anchor;
        polygon_centroids_with::<Z, M, MEASURE_Z, ANCHOR>(array, on_row_error)
    )
}

fn polygon_centroids_with<const Z: bool, const M: bool, const MEASURE_Z: bool, const ANCHOR: u8>(
    array: &Untrimmed<PolygonArray>,
    on_row_error: OnRowError,
) -> GeoComputeResult<PointArray> {
    centroids::<Z, M>(
        array.rows(),
        |i| array.row_vertices(i),
        Weighing::new(array.coords(), CentroidWeight::Area, MEASURE_Z),
        on_row_error,
        |moment, i| add_polygon_measured::<Z, M, MEASURE_Z, ANCHOR>(moment, &array.get(i)),
    )
}

pub fn multi_polygon_centroids(
    array: &Untrimmed<MultiPolygonArray>,
    method: CentroidMethod,
    anchor: CentroidAnchor,
    on_row_error: OnRowError,
) -> GeoComputeResult<PointArray> {
    let coords = array.coords();
    let dim = coords.dim();
    with_const_flags!(
        Z = dim.has_z(),
        M = dim.has_m(),
        MEASURE_Z = measures_z(method, coords),
        ANCHOR: CentroidAnchor = anchor;
        multi_polygon_centroids_with::<Z, M, MEASURE_Z, ANCHOR>(array, on_row_error)
    )
}

fn multi_polygon_centroids_with<
    const Z: bool,
    const M: bool,
    const MEASURE_Z: bool,
    const ANCHOR: u8,
>(
    array: &Untrimmed<MultiPolygonArray>,
    on_row_error: OnRowError,
) -> GeoComputeResult<PointArray> {
    centroids::<Z, M>(
        array.rows(),
        |i| array.row_vertices(i),
        Weighing::new(array.coords(), CentroidWeight::Area, MEASURE_Z),
        on_row_error,
        |moment, i| {
            for polygon in array.get(i).polygons() {
                add_polygon_measured::<Z, M, MEASURE_Z, ANCHOR>(moment, &polygon);
            }
        },
    )
}
