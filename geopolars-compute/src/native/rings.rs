//! The rings of a polygon:
//! - its exterior is its first ring, as a linestring,
//! - its interior is the rest (its holes), as a multilinestring.
//!
//! A multipolygon gets a list of them, one per polygon.
//!
//! # Const generic flags
//!
//! ## `ALL_VALID`
//!
//! [kept_rings_of_polygons] is generic over `ALL_VALID`.
//! If `ALL_VALID=true`, every row is there, so none is checked for being null.
//!
//! ## `Z`
//!
//! [close_open_rings] is generic over `Z`.
//! See [same_place] for `Z`.

use std::ops::Range;

use geopolars_core::output::list_array;
use geopolars_core::{
    same_place, LineStringArray, List, MultiLineStringArray, MultiPolygonArray, PolygonArray,
    Untrimmed, ValidCoords, ValidList,
};
use polars_arrow::array::ListArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::{Offsets, OffsetsBuffer};

use super::add_parts::{concatenated, Built, Nested};

/// Which of a polygon's rings `rings` to keep.
type Keep = fn(Range<usize>) -> Range<usize>;

#[inline(always)]
fn exterior_of(rings: Range<usize>) -> Range<usize> {
    rings.start..(rings.start + 1).min(rings.end)
}

#[inline(always)]
fn interior_of(rings: Range<usize>) -> Range<usize> {
    (rings.start + 1).min(rings.end)..rings.end
}

/// The rings every row of `array` keeps, none for a null row.
#[inline(always)]
fn kept_rings_of_polygons<const ALL_VALID: bool>(
    array: &Untrimmed<PolygonArray>,
    keep: Keep,
) -> impl Iterator<Item = Range<usize>> + Clone + '_ {
    let rows = array.rows();
    // A null row's offsets point anywhere, so it keeps nothing.
    (0..rows.len()).map(move |i| {
        if ALL_VALID || rows.is_valid(i) {
            keep(rows.range(i))
        } else {
            0..0
        }
    })
}

/// The rings every polygon of the rows of `array` keeps.
#[inline(always)]
fn kept_rings_of_multi_polygons(
    array: &Untrimmed<MultiPolygonArray>,
    keep: Keep,
) -> impl Iterator<Item = Range<usize>> + Clone + '_ {
    let polygons = array.polygons();
    array
        .rows()
        .span(0..array.rows().len())
        .map(move |j| keep(polygons.range(j)))
}

/// Offsets of lists of lengths `lengths`.
pub(super) fn offsets_of(lengths: impl Iterator<Item = usize>) -> OffsetsBuffer<i64> {
    Offsets::try_from_lengths(lengths)
        .expect("no more items than the input")
        .into()
}

/// One linestring per item of `kept` (a single ring or none),
/// with its row null where `validity` says.
#[inline(always)]
fn exteriors(
    rings: &ValidList,
    coords: &ValidCoords,
    kept: impl Iterator<Item = Range<usize>> + Clone,
    validity: Option<Bitmap>,
) -> LineStringArray {
    // The output cannot reuse the input's coordinate buffers.
    // In an Arrow list, each list ends where the next one starts,
    // meaning there is no way to leave out the dropped rings in between.
    // Instead, the coordinates of the kept rings are copied into new buffers.
    // The rings a polygon keeps (its exterior, or all of its holes) sit next to each other.
    // So for each polygon, its kept coordinates are copied as one slice per axis.
    // This is much faster than Polars' own `take` on the rings,
    // which copies ring by ring (~1.5 µs per ring).
    let spans = kept.map(|kept| rings.span(kept));
    LineStringArray::from_parts(
        List::from_offsets(offsets_of(spans.clone().map(|span| span.len())), validity),
        coords.copy_ranges(spans),
    )
}

/// One multilinestring per item of `kept`,
/// with its row null where `validity` says.
#[inline(always)]
fn interiors(
    rings: &ValidList,
    coords: &ValidCoords,
    kept: impl Iterator<Item = Range<usize>> + Clone,
    validity: Option<Bitmap>,
) -> MultiLineStringArray {
    let ring_lengths = kept
        .clone()
        .flat_map(|kept| kept.map(|ring| rings.range(ring).len()));
    MultiLineStringArray::from_parts(
        List::from_offsets(offsets_of(kept.clone().map(|kept| kept.len())), validity),
        ValidList::from_offsets(offsets_of(ring_lengths)),
        // Copied as in `exteriors`.
        coords.copy_ranges(kept.map(|kept| rings.span(kept))),
    )
}

/// `parts`, one list per row of `rows`.
fn per_multi_polygon(rows: &List, parts: ListArray<i64>) -> ListArray<i64> {
    let lengths = (0..rows.len()).map(|i| rows.range(i).len());
    list_array(parts.boxed(), offsets_of(lengths), rows.validity().cloned())
}

pub fn polygon_exteriors(array: &Untrimmed<PolygonArray>) -> LineStringArray {
    let validity = array.rows().validity().cloned();
    with_const_flags!(
        ALL_VALID = array.rows().is_all_valid();
        exteriors(
            array.rings(),
            array.coords(),
            kept_rings_of_polygons::<ALL_VALID>(array, exterior_of),
            validity,
        )
    )
}

pub fn polygon_interiors(array: &Untrimmed<PolygonArray>) -> MultiLineStringArray {
    let validity = array.rows().validity().cloned();
    with_const_flags!(
        ALL_VALID = array.rows().is_all_valid();
        interiors(
            array.rings(),
            array.coords(),
            kept_rings_of_polygons::<ALL_VALID>(array, interior_of),
            validity,
        )
    )
}

/// A list of linestrings per row, one per polygon.
pub fn multi_polygon_exteriors(array: &Untrimmed<MultiPolygonArray>) -> ListArray<i64> {
    let kept = kept_rings_of_multi_polygons(array, exterior_of);
    let parts = exteriors(array.rings(), array.coords(), kept, None);
    per_multi_polygon(array.rows(), parts.into_list_array())
}

/// A list of multilinestrings per row, one per polygon.
pub fn multi_polygon_interiors(array: &Untrimmed<MultiPolygonArray>) -> ListArray<i64> {
    let kept = kept_rings_of_multi_polygons(array, interior_of);
    let parts = interiors(array.rings(), array.coords(), kept, None);
    per_multi_polygon(array.rows(), parts.into_list_array())
}

// TODO: Not yet converted to bitmask pass, but is a minor candidate for it.
// Mark the rings that get their first vertex appended in one mask, and the rings that snap in another.
// `close_open_rings` then reads the masks, rather than calling `closing_of` again in every pass.
// With `snap_z=false` and `snap_m=false`, the first mask is the inverse of `rings_are_closed`.
/// What [close_open_rings] does with one ring.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Closing {
    Keep,
    AppendFirst,
    SnapLast,
}

#[inline(always)]
fn closing_of<const Z: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
    snap_z: bool,
    snap_m: bool,
) -> Closing {
    if vertices.is_empty() {
        return Closing::Keep;
    }
    let (first, last) = (vertices.start, vertices.end - 1);
    let differs = |axis: Option<&[f64]>| axis.is_some_and(|axis| axis[first] != axis[last]);
    let z_differs = Z && differs(coords.z());
    let m_differs = snap_m && differs(coords.m());
    if !same_place::<false>(coords, first, last) || (z_differs && !snap_z) {
        Closing::AppendFirst
    } else if z_differs || m_differs {
        Closing::SnapLast
    } else {
        Closing::Keep
    }
}

/// `axis`, with every ring of `rings` closed as `closing` says, in `len` values.
/// If `snaps=true`, a snapped ring's last value is replaced with its first.
fn closed_axis(
    axis: &[f64],
    rings: &ValidList,
    closing: &impl Fn(usize) -> Closing,
    snaps: bool,
    len: usize,
) -> Vec<f64> {
    let mut closed = Vec::with_capacity(len);
    for ring in 0..rings.len() {
        let vertices = rings.range(ring);
        closed.extend_from_slice(&axis[vertices.clone()]);
        match closing(ring) {
            Closing::AppendFirst => closed.push(axis[vertices.start]),
            Closing::SnapLast if snaps => {
                *closed.last_mut().expect("a snapped ring has vertices") = axis[vertices.start];
            }
            _ => {}
        }
    }
    closed
}

/// Closes every open ring of `rings`, so that its last vertex lies at the same place as its first.
/// `None` if no ring changes.
/// An empty ring is left empty.
///
/// A ring whose ends differ in x or y gets its first vertex repeated at its end.
/// So does a ring whose ends differ in z, unless `snap_z=true`.
/// If `snap_z=true`, its last vertex is moved onto the first one's z instead.
/// If `snap_m=true`, a ring whose ends differ in m gets its last vertex moved onto the first one's m,
/// unless its first vertex is repeated anyway.
/// If `snap_m=false`, m is not looked at.
/// A flag for an axis `coords` does not have does nothing.
pub fn close_open_rings<const Z: bool>(
    rings: &ValidList,
    coords: &ValidCoords,
    snap_z: bool,
    snap_m: bool,
) -> Option<LineStringArray> {
    // Checked again in every pass over the rings, rather than stored.
    let closing = |ring: usize| closing_of::<Z>(coords, rings.range(ring), snap_z, snap_m);
    if (0..rings.len()).all(|ring| closing(ring) == Closing::Keep) {
        return None;
    }
    let lengths = (0..rings.len())
        .map(|ring| rings.range(ring).len() + usize::from(closing(ring) == Closing::AppendFirst));
    let offsets = offsets_of(lengths);
    let len = *offsets.last() as usize;
    let closed = |axis: &[f64], snaps: bool| closed_axis(axis, rings, &closing, snaps, len);
    Some(LineStringArray::from_parts(
        List::from_offsets(offsets, None),
        ValidCoords::from_axes((
            closed(coords.x(), false),
            closed(coords.y(), false),
            coords.z().map(|z| closed(z, snap_z)),
            coords.m().map(|m| closed(m, snap_m)),
        )),
    ))
}

/// `polygons`, each followed by the linestrings on the same row of `holes` as extra holes.
/// A null row of `holes` adds no holes.
/// Without `holes`, the polygons are copied as they are.
/// The rows are null where `validity` says,
/// which has to cover the null rows of `polygons`.
///
/// `polygons` and `holes` have to have the same length and dimension.
/// The vertices are copied into new buffers, as one slice per axis per row of each.
pub fn polygons_with_added_holes(
    polygons: &PolygonArray,
    holes: Option<&MultiLineStringArray>,
    validity: Option<Bitmap>,
) -> PolygonArray {
    let Built {
        rows,
        levels: [rings],
        coords,
    } = concatenated(polygons.into(), holes.map(Nested::from), validity);
    PolygonArray::from_parts(rows, rings, coords)
}
