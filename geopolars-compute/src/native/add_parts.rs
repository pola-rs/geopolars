//! Adding the parts of one multi-geometry column to those of another,
//! and keeping only some of the parts of a multi-geometry.
//! Neither looks at the parts themselves: they are copied as they are.
//!
//! A polygon is a list of rings like a multilinestring is a list of linestrings,
//! so `polygons_with_added_holes` shares this machinery.

use std::array;
use std::ops::Range;

use geopolars_core::{
    List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray, ValidCoords,
    ValidList,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::Offsets;

const NO_OVERFLOW: &str = "no more items than the input";

/// A geometry array: its rows, the `N` list levels between them and the coordinates, outermost first,
/// and the coordinates.
pub(super) struct Nested<'a, const N: usize> {
    rows: &'a List,
    levels: [&'a ValidList; N],
    coords: &'a ValidCoords,
}

impl<'a> From<&'a MultiPointArray> for Nested<'a, 0> {
    fn from(array: &'a MultiPointArray) -> Self {
        Nested {
            rows: &array.rows,
            levels: [],
            coords: &array.coords,
        }
    }
}

impl<'a> From<&'a MultiLineStringArray> for Nested<'a, 1> {
    fn from(array: &'a MultiLineStringArray) -> Self {
        Nested {
            rows: &array.rows,
            levels: [&array.lines],
            coords: &array.coords,
        }
    }
}

impl<'a> From<&'a PolygonArray> for Nested<'a, 1> {
    fn from(array: &'a PolygonArray) -> Self {
        Nested {
            rows: &array.rows,
            levels: [&array.rings],
            coords: &array.coords,
        }
    }
}

impl<'a> From<&'a MultiPolygonArray> for Nested<'a, 2> {
    fn from(array: &'a MultiPolygonArray) -> Self {
        Nested {
            rows: &array.rows,
            levels: [&array.polygons, &array.rings],
            coords: &array.coords,
        }
    }
}

/// The parts of a [Nested] array, built up from copies of those of others.
pub(super) struct Built<const N: usize> {
    pub(super) rows: List,
    pub(super) levels: [ValidList; N],
    pub(super) coords: ValidCoords,
}

/// Extends `offsets` with those of the items `items` at every level below them,
/// and gives the range their coordinates take up.
#[inline(always)]
fn copied_offsets<const N: usize>(
    levels: &[&ValidList; N],
    mut items: Range<usize>,
    offsets: &mut [Offsets<i64>; N],
) -> Range<usize> {
    for (level, offsets) in levels.iter().zip(offsets) {
        // TODO: `try_extend_from_slice` is not inlined.
        // With only a few items per call, the call overhead is a large part of its cost.
        // It took ~5.5 ns/row of ~52 in a release profile of `add_holes`
        // (1M polygons with 1 exterior + 2-3 holes of 5 vertices).
        // An inlined loop of our own could win most of that back,
        // or we can contribute upstream?
        offsets
            .try_extend_from_slice(level.offsets(), items.start, items.len())
            .expect(NO_OVERFLOW);
        items = level.span(items);
    }
    items
}

fn offsets_with_capacity<const N: usize>(capacity: impl Fn(usize) -> usize) -> [Offsets<i64>; N] {
    array::from_fn(|level| Offsets::with_capacity(capacity(level)))
}

/// The vertices one valid row of [concatenated] copies.
struct CopiedVertices {
    /// In the coordinates of the base.
    base: Range<usize>,
    /// In the coordinates of the added parts.
    added: Range<usize>,
}

/// One axis of the base and of the added parts,
/// with every row's base followed by its added parts, in `len` values.
fn assembled_axis(base: &[f64], added: &[f64], copies: &[CopiedVertices], len: usize) -> Vec<f64> {
    let mut assembled = Vec::with_capacity(len);
    for copy in copies {
        assembled.extend_from_slice(&base[copy.base.clone()]);
        assembled.extend_from_slice(&added[copy.added.clone()]);
    }
    assembled
}

/// Every row of `base`, followed by the items on the same row of `added`.
/// A null row of `added` adds nothing.
/// Without `added`, `base` is copied as it is.
/// The rows are null where `validity` says,
/// which has to cover the null rows of `base`.
///
/// `base` and `added` have to have the same length and dimension.
/// The vertices are copied into new buffers, as one slice per axis per row of each.
pub(super) fn concatenated<const N: usize>(
    base: Nested<N>,
    added: Option<Nested<N>>,
    validity: Option<Bitmap>,
) -> Built<N> {
    let len = base.rows.len();
    if let Some(added) = &added {
        assert_eq!(added.rows.len(), len, "the arrays differ in length");
        assert_eq!(
            added.coords.dim(),
            base.coords.dim(),
            "the arrays differ in dimension"
        );
    }
    let is_valid = |i: usize| validity.as_ref().is_none_or(|validity| validity.get_bit(i));

    // One pass builds the offsets of every level, and the vertex ranges of each row.
    // The coordinates are copied per axis afterwards from those ranges.
    let added_items = |level: usize| added.as_ref().map_or(0, |added| added.levels[level].len());
    let mut row_offsets = Offsets::<i64>::with_capacity(len);
    let mut offsets = offsets_with_capacity(|level| base.levels[level].len() + added_items(level));
    let mut copies = Vec::with_capacity(len);
    let mut coords_len = 0;
    for i in 0..len {
        // A null row's offsets point anywhere, so it takes nothing.
        if !is_valid(i) {
            row_offsets.try_push(0).expect(NO_OVERFLOW);
            continue;
        }
        let items = base.rows.range(i);
        let mut item_count = items.len();
        let mut copy = CopiedVertices {
            base: copied_offsets(&base.levels, items, &mut offsets),
            added: 0..0,
        };
        if let Some(added) = added.as_ref().filter(|added| added.rows.is_valid(i)) {
            let items = added.rows.range(i);
            item_count += items.len();
            copy.added = copied_offsets(&added.levels, items, &mut offsets);
        }
        row_offsets.try_push(item_count).expect(NO_OVERFLOW);
        coords_len += copy.base.len() + copy.added.len();
        copies.push(copy);
    }

    let (from, to) = (base.coords, added.as_ref().map(|added| added.coords));
    let axis = |base: &[f64], added: Option<&[f64]>| {
        assembled_axis(base, added.unwrap_or_default(), &copies, coords_len)
    };
    Built {
        rows: List::from_offsets(row_offsets.into(), validity),
        levels: offsets.map(|offsets| ValidList::from_offsets(offsets.into())),
        coords: ValidCoords::from_axes((
            axis(from.x(), to.map(ValidCoords::x)),
            axis(from.y(), to.map(ValidCoords::y)),
            from.z().map(|z| axis(z, to.and_then(ValidCoords::z))),
            from.m().map(|m| axis(m, to.and_then(ValidCoords::m))),
        )),
    }
}

/// The runs of consecutive items of `items` set in `keep`.
fn kept_runs(keep: &Bitmap, items: Range<usize>) -> impl Iterator<Item = Range<usize>> + '_ {
    let end = items.end;
    let mut next = items.start;
    std::iter::from_fn(move || {
        let start = (next..end).find(|&j| keep.get_bit(j))?;
        let stop = (start..end).find(|&j| !keep.get_bit(j)).unwrap_or(end);
        next = stop;
        Some(start..stop)
    })
}

/// `array` with only the items set in `keep` (one bit per item),
/// or `None` when that is all of them.
/// The rows keep their validity.
fn kept<const N: usize>(array: Nested<N>, keep: &Bitmap) -> Option<Built<N>> {
    if keep.unset_bits() == 0 {
        return None;
    }
    let len = array.rows.len();
    let mut row_offsets = Offsets::<i64>::with_capacity(len);
    let mut offsets = offsets_with_capacity(|level| array.levels[level].len());
    let mut vertices = Vec::new();
    for i in 0..len {
        // A null row's offsets point anywhere, so it takes nothing.
        if !array.rows.is_valid(i) {
            row_offsets.try_push(0).expect(NO_OVERFLOW);
            continue;
        }
        let mut item_count = 0;
        for run in kept_runs(keep, array.rows.range(i)) {
            item_count += run.len();
            vertices.push(copied_offsets(&array.levels, run, &mut offsets));
        }
        row_offsets.try_push(item_count).expect(NO_OVERFLOW);
    }
    Some(Built {
        rows: List::from_offsets(row_offsets.into(), array.rows.validity().cloned()),
        levels: offsets.map(|offsets| ValidList::from_offsets(offsets.into())),
        coords: array.coords.copy_ranges(vertices.into_iter()),
    })
}

/// `base`, each followed by the points on the same row of `added`.
/// See [multi_line_strings_with_added_parts].
pub fn multi_points_with_added_parts(
    base: &MultiPointArray,
    added: &MultiPointArray,
    validity: Option<Bitmap>,
) -> MultiPointArray {
    let Built { rows, coords, .. } = concatenated(base.into(), Some(added.into()), validity);
    MultiPointArray::from_parts(rows, coords)
}

/// `base`, each followed by the linestrings on the same row of `added`.
/// A null row of `added` adds nothing.
/// The rows are null where `validity` says,
/// which has to cover the null rows of `base`.
///
/// `base` and `added` have to have the same length and dimension.
pub fn multi_line_strings_with_added_parts(
    base: &MultiLineStringArray,
    added: &MultiLineStringArray,
    validity: Option<Bitmap>,
) -> MultiLineStringArray {
    let Built {
        rows,
        levels: [lines],
        coords,
    } = concatenated(base.into(), Some(added.into()), validity);
    MultiLineStringArray::from_parts(rows, lines, coords)
}

/// `base`, each followed by the polygons on the same row of `added`.
/// See [multi_line_strings_with_added_parts].
pub fn multi_polygons_with_added_parts(
    base: &MultiPolygonArray,
    added: &MultiPolygonArray,
    validity: Option<Bitmap>,
) -> MultiPolygonArray {
    let Built {
        rows,
        levels: [polygons, rings],
        coords,
    } = concatenated(base.into(), Some(added.into()), validity);
    MultiPolygonArray::from_parts(rows, polygons, rings, coords)
}

/// `array` with only the points set in `keep`, one bit per point.
/// Free when that is all of them.
pub fn multi_points_with_kept_parts(array: MultiPointArray, keep: &Bitmap) -> MultiPointArray {
    match kept((&array).into(), keep) {
        Some(Built { rows, coords, .. }) => MultiPointArray::from_parts(rows, coords),
        None => array,
    }
}

/// `array` with only the linestrings set in `keep`, one bit per linestring.
/// Free when that is all of them.
pub fn multi_line_strings_with_kept_parts(
    array: MultiLineStringArray,
    keep: &Bitmap,
) -> MultiLineStringArray {
    match kept((&array).into(), keep) {
        Some(Built {
            rows,
            levels: [lines],
            coords,
        }) => MultiLineStringArray::from_parts(rows, lines, coords),
        None => array,
    }
}

/// `array` with only the polygons set in `keep`, one bit per polygon.
/// Free when that is all of them.
pub fn multi_polygons_with_kept_parts(
    array: MultiPolygonArray,
    keep: &Bitmap,
) -> MultiPolygonArray {
    match kept((&array).into(), keep) {
        Some(Built {
            rows,
            levels: [polygons, rings],
            coords,
        }) => MultiPolygonArray::from_parts(rows, polygons, rings, coords),
        None => array,
    }
}
