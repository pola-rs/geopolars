//! Copying the parts on every row of one array after those of another.
//! `add_parts` builds on it, and so does `polygons_with_added_holes`,
//! as a polygon's rings are its parts like a multilinestring's linestrings.

use std::ops::Range;
use std::{array, mem};

use geopolars_core::{
    List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray, ValidCoords,
    ValidList,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::Offsets;

const NO_OVERFLOW: &str = "no more items than the input";

/// A geometry array as its rows, the `N` list levels below them (outermost first),
/// and its coordinates.
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

/// The rows, list levels, and coordinates [concatenated] builds.
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
        // PERF: `try_extend_from_slice` is not inlined.
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

/// A slice of the base's vertices followed by a slice of the added parts' vertices,
/// as [concatenated] copies them.
/// A valid row takes one, and one more for each run of kept items after the first (see [kept_runs]).
/// The last run of the base and the first run of the added parts share one.
/// Every other run has one to itself, with the other range empty.
#[derive(Default)]
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
/// The items are the entries of a row's list:
/// the parts of a multi-geometry, or the rings of a polygon.
/// A null row of `added` adds nothing.
/// With `base_keep` or `added_keep`, only the items of `base` or `added` set in it are taken
/// (one bit per item).
/// Without `added`, `base` is copied as it is (apart from what `base_keep` leaves out).
/// The rows are null where `validity` says.
/// Every null row of `base` has to be null in it too.
///
/// `base` and `added` have to have the same length and dimension.
/// The vertices are copied into new buffers, two slices per axis per [CopiedVertices].
pub(super) fn concatenated<const N: usize>(
    base: Nested<N>,
    base_keep: Option<&Bitmap>,
    added: Option<Nested<N>>,
    added_keep: Option<&Bitmap>,
    validity: Option<Bitmap>,
) -> Built<N> {
    let base_keep = base_keep.filter(|keep| keep.unset_bits() > 0);
    let added_keep = added_keep.filter(|keep| keep.unset_bits() > 0);
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

    // One pass builds the offsets of every level, and the vertex ranges to copy.
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
        let mut item_count = 0;
        let mut copy = CopiedVertices::default();
        match base_keep {
            None => {
                item_count = items.len();
                copy.base = copied_offsets(&base.levels, items, &mut offsets);
            }
            // A single item, as every row of a single geometry has, needs no search for runs.
            Some(keep) if items.len() == 1 => {
                if keep.get_bit(items.start) {
                    item_count = 1;
                    copy.base = copied_offsets(&base.levels, items, &mut offsets);
                }
            }
            // As for the added items below, every run is a slice of its own.
            // The last run shares its `CopiedVertices` with the first run of the added items.
            // Nothing reaches this yet, so it is untested:
            // `add_parts` only masks a single base, of one item per row.
            // It would be reached by dropping the empty parts of a multi base in `add_parts`,
            // or by an expression that drops parts of one column, without `added`.
            Some(keep) => {
                for run in kept_runs(keep, items) {
                    if item_count > 0 {
                        coords_len += copy.base.len();
                        copies.push(mem::take(&mut copy));
                    }
                    item_count += run.len();
                    copy.base = copied_offsets(&base.levels, run, &mut offsets);
                }
            }
        }
        let added = added.as_ref().filter(|added| added.rows.is_valid(i));
        let Some(added) = added else {
            row_offsets.try_push(item_count).expect(NO_OVERFLOW);
            coords_len += copy.base.len();
            copies.push(copy);
            continue;
        };
        let Some(keep) = added_keep else {
            let items = added.rows.range(i);
            item_count += items.len();
            copy.added = copied_offsets(&added.levels, items, &mut offsets);
            row_offsets.try_push(item_count).expect(NO_OVERFLOW);
            coords_len += copy.base.len() + copy.added.len();
            copies.push(copy);
            continue;
        };
        // The vertices of the items unset in `keep` lie between the runs, and must not be copied.
        // So every run is a slice of its own.
        // The first run shares the base's `CopiedVertices`, every later run gets one of its own.
        let mut runs = kept_runs(keep, added.rows.range(i));
        if let Some(run) = runs.next() {
            item_count += run.len();
            copy.added = copied_offsets(&added.levels, run, &mut offsets);
        }
        coords_len += copy.base.len() + copy.added.len();
        copies.push(copy);
        for run in runs {
            item_count += run.len();
            let added_vertices = copied_offsets(&added.levels, run, &mut offsets);
            coords_len += added_vertices.len();
            copies.push(CopiedVertices {
                base: 0..0,
                added: added_vertices,
            });
        }
        row_offsets.try_push(item_count).expect(NO_OVERFLOW);
    }

    let (base_coords, added_coords) = (base.coords, added.as_ref().map(|added| added.coords));
    let axis = |base: &[f64], added: Option<&[f64]>| {
        assembled_axis(base, added.unwrap_or_default(), &copies, coords_len)
    };
    Built {
        rows: List::from_offsets(row_offsets.into(), validity),
        levels: offsets.map(|offsets| ValidList::from_offsets(offsets.into())),
        coords: ValidCoords::from_axes((
            axis(base_coords.x(), added_coords.map(ValidCoords::x)),
            axis(base_coords.y(), added_coords.map(ValidCoords::y)),
            (base_coords.z()).map(|z| axis(z, added_coords.and_then(ValidCoords::z))),
            (base_coords.m()).map(|m| axis(m, added_coords.and_then(ValidCoords::m))),
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
