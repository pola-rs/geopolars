//! Adding the parts of one multi-geometry column to those of another.
//! The parts are not looked at, only copied as they are.
//!
//! A polygon is a list of rings like a multilinestring is a list of linestrings,
//! so `polygons_with_added_holes` shares this machinery.

use std::array;
use std::ops::Range;

use geopolars_core::{
    Coords, LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray, ValidCoords, ValidList,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::Offsets;

use super::is_empty::{line_strings_are_empty, points_are_empty, polygons_are_empty};

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
/// With `keep`, only the items of `added` set in it are added (one bit per item).
/// Without `added`, `base` is copied as it is.
/// The rows are null where `validity` says,
/// which has to cover the null rows of `base`.
///
/// `base` and `added` have to have the same length and dimension.
/// The vertices are copied into new buffers, as one slice per axis per row of each.
pub(super) fn concatenated<const N: usize>(
    base: Nested<N>,
    added: Option<Nested<N>>,
    keep: Option<&Bitmap>,
    validity: Option<Bitmap>,
) -> Built<N> {
    let keep = keep.filter(|keep| keep.unset_bits() > 0);
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
        let added = added.as_ref().filter(|added| added.rows.is_valid(i));
        let Some(added) = added else {
            row_offsets.try_push(item_count).expect(NO_OVERFLOW);
            coords_len += copy.base.len();
            copies.push(copy);
            continue;
        };
        let Some(keep) = keep else {
            let items = added.rows.range(i);
            item_count += items.len();
            copy.added = copied_offsets(&added.levels, items, &mut offsets);
            row_offsets.try_push(item_count).expect(NO_OVERFLOW);
            coords_len += copy.base.len() + copy.added.len();
            copies.push(copy);
            continue;
        };
        // The first run goes with the base, every later run is a copy of its own.
        let mut runs = kept_runs(keep, added.rows.range(i));
        if let Some(run) = runs.next() {
            item_count += run.len();
            copy.added = copied_offsets(&added.levels, run, &mut offsets);
        }
        coords_len += copy.base.len() + copy.added.len();
        copies.push(copy);
        for run in runs {
            item_count += run.len();
            let added = copied_offsets(&added.levels, run, &mut offsets);
            coords_len += added.len();
            copies.push(CopiedVertices { base: 0..0, added });
        }
        row_offsets.try_push(item_count).expect(NO_OVERFLOW);
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

/// A view whose parts may hold null values, and a mask of the parts that do not.
/// A kernel taking it reads only the parts set in the mask,
/// so the view is not handed out as a plain one.
pub struct NullPartsMarked<T> {
    pub(crate) view: T,
    /// One bit per part. `None` when every part is set.
    pub(crate) passing: Option<Bitmap>,
}

/// A plain view has no null values below its rows, so every part is set.
impl<T> From<T> for NullPartsMarked<T> {
    fn from(view: T) -> Self {
        Self {
            view,
            passing: None,
        }
    }
}

/// `passing` with the parts `empty` holds as `true` unset too.
fn passing_without_empty(passing: Option<Bitmap>, empty: BooleanArray) -> Option<Bitmap> {
    combine_validities_and(passing.as_ref(), Some(&!empty.values()))
}

impl NullPartsMarked<MultiPointArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.view.rows
    }

    /// Unsets the empty points (every axis NaN) in the mask too.
    pub fn without_empty_parts(self) -> Self {
        let points = PointArray::new(Coords::from_valid(self.view.coords.clone(), None));
        let passing = passing_without_empty(self.passing, points_are_empty(&points));
        Self { passing, ..self }
    }
}

impl NullPartsMarked<MultiLineStringArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.view.rows
    }

    /// Unsets the linestrings without vertices in the mask too.
    pub fn without_empty_parts(self) -> Self {
        let lines = LineStringArray::from_parts(
            List::from_offsets(self.view.lines.offsets().clone(), None),
            self.view.coords.clone(),
        );
        let empty = line_strings_are_empty::<false>(&lines.into());
        let passing = passing_without_empty(self.passing, empty);
        Self { passing, ..self }
    }
}

impl NullPartsMarked<MultiPolygonArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.view.rows
    }

    /// Unsets the polygons without vertices in the mask too.
    pub fn without_empty_parts(self) -> Self {
        let polygons = PolygonArray::from_parts(
            List::from_offsets(self.view.polygons.offsets().clone(), None),
            self.view.rings.clone(),
            self.view.coords.clone(),
        );
        let empty = polygons_are_empty::<false>(&polygons.into());
        let passing = passing_without_empty(self.passing, empty);
        Self { passing, ..self }
    }
}

/// `base`, each followed by the points on the same row of `added`.
/// See [multi_line_strings_with_added_parts].
pub fn multi_points_with_added_parts(
    base: &MultiPointArray,
    added: &NullPartsMarked<MultiPointArray>,
    validity: Option<Bitmap>,
) -> MultiPointArray {
    let keep = added.passing.as_ref();
    let Built { rows, coords, .. } =
        concatenated(base.into(), Some((&added.view).into()), keep, validity);
    MultiPointArray::from_parts(rows, coords)
}

/// `base`, each followed by the linestrings on the same row of `added`.
/// A null row of `added` adds nothing.
/// Only the linestrings of `added` set in its mask are added.
/// The rows are null where `validity` says,
/// which has to cover the null rows of `base`.
///
/// `base` and `added` have to have the same length and dimension.
pub fn multi_line_strings_with_added_parts(
    base: &MultiLineStringArray,
    added: &NullPartsMarked<MultiLineStringArray>,
    validity: Option<Bitmap>,
) -> MultiLineStringArray {
    let keep = added.passing.as_ref();
    let Built {
        rows,
        levels: [lines],
        coords,
    } = concatenated(base.into(), Some((&added.view).into()), keep, validity);
    MultiLineStringArray::from_parts(rows, lines, coords)
}

/// `base`, each followed by the polygons on the same row of `added`.
/// See [multi_line_strings_with_added_parts].
pub fn multi_polygons_with_added_parts(
    base: &MultiPolygonArray,
    added: &NullPartsMarked<MultiPolygonArray>,
    validity: Option<Bitmap>,
) -> MultiPolygonArray {
    let keep = added.passing.as_ref();
    let Built {
        rows,
        levels: [polygons, rings],
        coords,
    } = concatenated(base.into(), Some((&added.view).into()), keep, validity);
    MultiPolygonArray::from_parts(rows, polygons, rings, coords)
}
