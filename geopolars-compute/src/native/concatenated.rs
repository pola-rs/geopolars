//! Per-row concatenation of parts onto geometries.

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

/// The rows, list levels, and coordinates [concatenate_parts_onto_geometry] builds.
pub(super) struct Built<const N: usize> {
    pub(super) rows: List,
    pub(super) levels: [ValidList; N],
    pub(super) coords: ValidCoords,
}

/// Copies the offsets of `items`, and of every level below them, onto `offsets`.
/// Returns the range of their vertices.
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

#[derive(Default)]
struct CopiedVertices {
    /// In the coordinates of the base.
    base: Range<usize>,
    /// In the coordinates of the added parts.
    added: Range<usize>,
}

/// Assembles one axis from the vertex ranges in `copies`.
/// Each copy takes its range of `base`, then its range of `added`.
fn assembled_axis(
    base: &[f64],
    added: &[f64],
    copies: &[CopiedVertices],
    assembled_len: usize,
) -> Vec<f64> {
    let mut assembled = Vec::with_capacity(assembled_len);
    for copy in copies {
        assembled.extend_from_slice(&base[copy.base.clone()]);
        assembled.extend_from_slice(&added[copy.added.clone()]);
    }
    assembled
}

/// Performs a per-row concatenation of parts
/// (the rings of a polygon, or the linestrings of a multilinestring)
/// onto a base geometry.
/// Only the parts set in `base_keep` and `parts_keep` are kept,
/// so without `parts_added` this only filters `base_geometry`.
///
/// `base_geometry` and `parts_added` must have the same length and dimension.
/// `output_validity` must mark every null row of `base_geometry` as null.
pub(super) fn concatenate_parts_onto_geometry<const N: usize>(
    base_geometry: Nested<N>,
    base_keep: Option<&Bitmap>,
    parts_added: Option<Nested<N>>,
    parts_keep: Option<&Bitmap>,
    output_validity: Option<Bitmap>,
) -> Built<N> {
    let base_keep = base_keep.filter(|keep| keep.unset_bits() > 0);
    let parts_keep = parts_keep.filter(|keep| keep.unset_bits() > 0);
    let len = base_geometry.rows.len();
    if let Some(added) = &parts_added {
        assert_eq!(added.rows.len(), len, "the arrays differ in length");
        assert_eq!(
            added.coords.dim(),
            base_geometry.coords.dim(),
            "the arrays differ in dimension"
        );
    }
    let is_valid = |i: usize| {
        output_validity
            .as_ref()
            .is_none_or(|validity| validity.get_bit(i))
    };

    // One pass builds the offsets of every level, and the vertex ranges to copy.
    // The coordinates are copied per axis afterwards from those ranges.
    let added_items = |level: usize| {
        parts_added
            .as_ref()
            .map_or(0, |added| added.levels[level].len())
    };
    let mut row_offsets = Offsets::<i64>::with_capacity(len);
    let mut offsets =
        offsets_with_capacity(|level| base_geometry.levels[level].len() + added_items(level));
    let mut copies = Vec::with_capacity(len);
    let mut coords_len = 0;
    for i in 0..len {
        // A null row's offsets point anywhere, so it takes nothing.
        if !is_valid(i) {
            row_offsets.try_push(0).expect(NO_OVERFLOW);
            continue;
        }
        let items = base_geometry.rows.range(i);
        let mut item_count = 0;
        let mut copy = CopiedVertices::default();
        match base_keep {
            None => {
                item_count = items.len();
                copy.base = copied_offsets(&base_geometry.levels, items, &mut offsets);
            }
            // A single item, as every row of a single geometry has, needs no search for runs.
            Some(keep) if items.len() == 1 => {
                if keep.get_bit(items.start) {
                    item_count = 1;
                    copy.base = copied_offsets(&base_geometry.levels, items, &mut offsets);
                }
            }
            // As for the added items below, every run is a slice of its own.
            // The last run shares its `CopiedVertices` with the first run of the added items.
            // Untested, as nothing masks a base of more than one item per row yet.
            // It is there for dropping parts of a multi base, such as its empty ones.
            Some(keep) => {
                for run in kept_runs(keep, items) {
                    if item_count > 0 {
                        coords_len += copy.base.len();
                        copies.push(mem::take(&mut copy));
                    }
                    item_count += run.len();
                    copy.base = copied_offsets(&base_geometry.levels, run, &mut offsets);
                }
            }
        }
        let added = parts_added.as_ref().filter(|added| added.rows.is_valid(i));
        let Some(added) = added else {
            row_offsets.try_push(item_count).expect(NO_OVERFLOW);
            coords_len += copy.base.len();
            copies.push(copy);
            continue;
        };
        let Some(keep) = parts_keep else {
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

    let base_coords = base_geometry.coords;
    let added_coords = parts_added.as_ref().map(|added| added.coords);
    let axis = |base: &[f64], added: Option<&[f64]>| {
        assembled_axis(base, added.unwrap_or_default(), &copies, coords_len)
    };
    Built {
        rows: List::from_offsets(row_offsets.into(), output_validity),
        levels: offsets.map(|offsets| ValidList::from_offsets(offsets.into())),
        coords: ValidCoords::from_axes((
            axis(base_coords.x(), added_coords.map(ValidCoords::x)),
            axis(base_coords.y(), added_coords.map(ValidCoords::y)),
            (base_coords.z()).map(|z| axis(z, added_coords.and_then(ValidCoords::z))),
            (base_coords.m()).map(|m| axis(m, added_coords.and_then(ValidCoords::m))),
        )),
    }
}

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
