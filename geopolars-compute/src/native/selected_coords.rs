//! What to do with the selected coordinates of a geometry.
//! Implements the `action` and `short_lines` of the Python `filter_coordinates` and `fix_out_of_bounds`.
//! Every kernel marks the coordinates its [CoordinateSelection] selects ([selected_coords]),
//! then takes its [SelectedAction] on them.
//!
//! [SelectedAction::Drop] copies the kept coordinates once, a run of them at a time.
//! No part is ever removed, so only the innermost list level gets new offsets.
//! A null row is processed like any other, as its value is masked anyway.
//!
//! # Const generic flags
//!
//! ## `CHECK_CLOSED`
//!
//! [without_selected] is generic over `CHECK_CLOSED`.
//! If `CHECK_CLOSED=true`, a ring that is not closed after the drop is too short.
//! If `CHECK_CLOSED=false`, the caller guarantees each ring keeps both its ends,
//! so it is as closed as it came in.
//! Only the kernels of rings dispatch on it.
//!
//! ## `Z`
//!
//! [without_selected] is generic over `Z`, to check whether a ring is still closed.
//! See [same_place] for `Z`.
//! Only the kernels of rings with `CHECK_CLOSED=true` dispatch on it.

use std::iter::Peekable;
use std::ops::Range;

use geopolars_views::{
    same_place, Coords, LineStringArray, List, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, ValidCoords, ValidList, MIN_RING_VERTICES,
};
use polars_arrow::bitmap::iterator::ExactTrueIdxIter;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::OffsetsBuffer;

use super::coordinate_selection::{selected_coords, which_are_not_selected, CoordinateSelection};
use super::rings::offsets_of;
use crate::passing::valid_rows_passing;
use crate::utils::copy_ranges;
use crate::ShortLines;

/// See `action` of `NonFinite` and `OutOfBounds` in Python.
#[derive(Debug, Clone, Copy)]
pub enum SelectedAction {
    SetEmpty,
    Drop,
    SetNull,
}

/// [selected_coords], with no bit set when `selection` selects nothing.
fn selected_coords_bitmap(coords: &ValidCoords, selection: &CoordinateSelection) -> Bitmap {
    selected_coords(coords, selection).unwrap_or_else(|| Bitmap::new_zeroed(coords.len()))
}

/// `coords`, with every axis NaN where `selected`.
fn emptied(coords: &ValidCoords, selected: &Bitmap) -> ValidCoords {
    let emptied = |axis: &[f64]| -> Vec<f64> {
        axis.iter()
            .zip(selected.iter())
            .map(|(&value, selected)| if selected { f64::NAN } else { value })
            .collect()
    };
    ValidCoords::from_axes((
        emptied(coords.x()),
        emptied(coords.y()),
        coords.z().map(emptied),
        coords.m().map(emptied),
    ))
}

/// What the lists of vertices handed to [without_selected] are.
#[derive(Clone, Copy)]
enum Lists {
    /// The points of a multipoint, which are never too short.
    Points,
    Lines,
    Rings,
}

/// Pushes the runs of unselected vertices in `vertices` onto `runs`, and returns how many vertices they hold.
///
/// As it takes every vertex before `vertices.end` from `selected`,
/// it must be called on every list in turn, with the same `selected`.
#[inline(always)]
fn push_kept_runs(
    selected: &mut Peekable<ExactTrueIdxIter<'_>>,
    vertices: Range<usize>,
    runs: &mut Vec<Range<usize>>,
) -> usize {
    let (mut kept, mut run_start) = (vertices.len(), vertices.start);
    while let Some(i) = selected.next_if(|&i| i < vertices.end) {
        if run_start < i {
            runs.push(run_start..i);
        }
        run_start = i + 1;
        kept -= 1;
    }
    if run_start < vertices.end {
        runs.push(run_start..vertices.end);
    }
    kept
}

/// The most coordinates [merged_where_contiguous] merges runs up to.
/// `copy_ranges` copies a run one axis at a time, so this is 2 KB per copy.
/// From about 4 KB on (depending on the CPU), glibc's `memcpy` switches to `rep movsb`.
/// That is slow here: after a dropped vertex, the copy's destination is no longer aligned like its source.
const MAX_MERGED_RUN: usize = 256;

/// `runs`, with neighbouring runs merged: one that starts where the previous one ends is joined to it.
/// A merged run holds at most [MAX_MERGED_RUN] coordinates; a longer run stays as it is.
/// Most lists are kept whole, so this turns their runs into far fewer copies.
fn merged_where_contiguous(
    runs: impl Iterator<Item = Range<usize>> + Clone,
) -> impl Iterator<Item = Range<usize>> + Clone {
    let mut runs = runs.peekable();
    std::iter::from_fn(move || {
        let mut merged = runs.next()?;
        while let Some(next) = runs
            .next_if(|next| next.start == merged.end && next.end - merged.start <= MAX_MERGED_RUN)
        {
            merged.end = next.end;
        }
        Some(merged)
    })
}

/// Whether the `kept` vertices in `runs` are too short for `lists`:
/// a linestring of a single vertex,
/// or a ring with vertices that has fewer than [MIN_RING_VERTICES], or is not closed.
#[inline(always)]
fn is_too_short<const CHECK_CLOSED: bool, const Z: bool>(
    coords: &ValidCoords,
    lists: Lists,
    runs: &[Range<usize>],
    kept: usize,
) -> bool {
    match lists {
        Lists::Points => false,
        Lists::Lines => kept == 1,
        Lists::Rings => match (runs.first(), runs.last()) {
            (Some(first), Some(last)) => {
                kept < MIN_RING_VERTICES
                    || (CHECK_CLOSED && !same_place::<Z>(coords, first.start, last.end - 1))
            }
            _ => false,
        },
    }
}

/// Drops the `selected` vertices from every one of `lists`,
/// and handles a list that is then [too short](is_too_short) as `short_lines` says.
/// Returns the new offsets and coordinates, with a mask of the lists that pass.
/// Only a list that is too short with [ShortLines::Null] fails, and it keeps no vertices.
fn without_selected<const CHECK_CLOSED: bool, const Z: bool>(
    lists: &ValidList,
    kind: Lists,
    coords: &ValidCoords,
    selected: &Bitmap,
    short_lines: ShortLines,
) -> (OffsetsBuffer<i64>, ValidCoords, Option<Bitmap>) {
    debug_assert!(
        lists.spans_all_of(coords.len()),
        "the lists have to be trimmed (see \"Trimming\" in docs/source/development/geo/view-types.md)"
    );
    let (mut runs, mut passing) = (Vec::new(), None);
    let mut selected = selected.true_idx_iter().peekable();
    let lengths: Vec<usize> = lists
        .ranges()
        .enumerate()
        .map(|(j, list)| {
            let first_run = runs.len();
            let kept = push_kept_runs(&mut selected, list, &mut runs);
            if !is_too_short::<CHECK_CLOSED, Z>(coords, kind, &runs[first_run..], kept) {
                return kept;
            }
            match short_lines {
                ShortLines::Keep => kept,
                ShortLines::Empty => {
                    runs.truncate(first_run);
                    0
                }
                ShortLines::Null => {
                    runs.truncate(first_run);
                    passing
                        .get_or_insert_with(|| MutableBitmap::from_len_set(lists.len()))
                        .set(j, false);
                    0
                }
            }
        })
        .collect();
    let passing = passing.map(Bitmap::from);
    // Every list kept every vertex, so the coordinates and offsets are shared rather than copied.
    if lengths.iter().sum::<usize>() == coords.len() {
        return (lists.offsets().clone(), coords.clone(), passing);
    }
    (
        offsets_of(lengths.into_iter()),
        copy_ranges(coords, merged_where_contiguous(runs.iter().cloned())),
        passing,
    )
}

/// [without_selected] for the rings of `coords`.
fn without_selected_in_rings(
    rings: &ValidList,
    coords: &ValidCoords,
    selected: &Bitmap,
    short_lines: ShortLines,
    check_closed: bool,
) -> (OffsetsBuffer<i64>, ValidCoords, Option<Bitmap>) {
    let kind = Lists::Rings;
    // `Z` only matters for the closure check, so it is not dispatched on without it.
    if check_closed {
        with_const_flags!(
            Z = coords.dim().has_z();
            without_selected::<true, Z>(rings, kind, coords, selected, short_lines)
        )
    } else {
        without_selected::<false, false>(rings, kind, coords, selected, short_lines)
    }
}

pub(super) fn line_strings_without_selected(
    array: &LineStringArray,
    selected: &Bitmap,
    short_lines: ShortLines,
) -> LineStringArray {
    let (offsets, coords, passing) = without_selected::<false, false>(
        array.rows.ignoring_validity(),
        Lists::Lines,
        &array.coords,
        selected,
        short_lines,
    );
    let validity = combine_validities_and(array.rows.validity(), passing.as_ref());
    LineStringArray::from_parts(List::from_offsets(offsets, validity), coords)
}

pub(super) fn multi_points_without_selected(
    array: &MultiPointArray,
    selected: &Bitmap,
    short_lines: ShortLines,
) -> MultiPointArray {
    let (offsets, coords, _) = without_selected::<false, false>(
        array.rows.ignoring_validity(),
        Lists::Points,
        &array.coords,
        selected,
        short_lines,
    );
    MultiPointArray::from_parts(
        List::from_offsets(offsets, array.rows.validity().cloned()),
        coords,
    )
}

pub(super) fn polygons_without_selected(
    array: &PolygonArray,
    selected: &Bitmap,
    short_lines: ShortLines,
    check_closed: bool,
) -> PolygonArray {
    let (rings, coords, passing) = without_selected_in_rings(
        &array.rings,
        &array.coords,
        selected,
        short_lines,
        check_closed,
    );
    PolygonArray::from_parts(
        rows_nulled_unless_passing(&array.rows, &[], passing),
        ValidList::from_offsets(rings),
        coords,
    )
}

pub(super) fn multi_line_strings_without_selected(
    array: &MultiLineStringArray,
    selected: &Bitmap,
    short_lines: ShortLines,
) -> MultiLineStringArray {
    let (lines, coords, passing) = without_selected::<false, false>(
        &array.lines,
        Lists::Lines,
        &array.coords,
        selected,
        short_lines,
    );
    MultiLineStringArray::from_parts(
        rows_nulled_unless_passing(&array.rows, &[], passing),
        ValidList::from_offsets(lines),
        coords,
    )
}

pub(super) fn multi_polygons_without_selected(
    array: &MultiPolygonArray,
    selected: &Bitmap,
    short_lines: ShortLines,
    check_closed: bool,
) -> MultiPolygonArray {
    let (rings, coords, passing) = without_selected_in_rings(
        &array.rings,
        &array.coords,
        selected,
        short_lines,
        check_closed,
    );
    MultiPolygonArray::from_parts(
        rows_nulled_unless_passing(&array.rows, &[&array.polygons], passing),
        array.polygons.clone(),
        ValidList::from_offsets(rings),
        coords,
    )
}

/// `rows`, null too where a row holds an item that fails `passing` (see [valid_rows_passing]).
/// `inner_lists` are the list levels below the rows, outermost first.
fn rows_nulled_unless_passing(
    rows: &List,
    inner_lists: &[&ValidList],
    passing: Option<Bitmap>,
) -> List {
    rows.clone()
        .with_validity(valid_rows_passing(rows, inner_lists, passing))
}

/// `rows`, null too where a row holds a coordinate `selection` selects.
/// `inner_lists` are the list levels below the rows, outermost first.
fn rows_nulled_where_selected(
    rows: &List,
    inner_lists: &[&ValidList],
    coords: &ValidCoords,
    selection: &CoordinateSelection,
) -> List {
    let not_selected = which_are_not_selected(coords, selection);
    rows_nulled_unless_passing(rows, inner_lists, not_selected)
}

/// [SelectedAction::Drop] leaves the empty point, as a point cannot lose its coordinate.
pub fn points_with_selected(
    array: &PointArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
) -> PointArray {
    let (coords, validity) = (array.coords.ignoring_validity(), array.coords.validity());
    PointArray::new(match action {
        SelectedAction::SetEmpty | SelectedAction::Drop => {
            let selected = &selected_coords_bitmap(coords, selection);
            Coords::from_valid(emptied(coords, selected), validity.cloned())
        }
        SelectedAction::SetNull => {
            let not_selected = which_are_not_selected(coords, selection);
            let validity = combine_validities_and(validity, not_selected.as_ref());
            Coords::from_valid(coords.clone(), validity)
        }
    })
}

pub fn line_strings_with_selected(
    array: &LineStringArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
    short_lines: ShortLines,
) -> LineStringArray {
    let selected = || selected_coords_bitmap(&array.coords, selection);
    match action {
        SelectedAction::SetEmpty => {
            LineStringArray::from_parts(array.rows.clone(), emptied(&array.coords, &selected()))
        }
        SelectedAction::Drop => line_strings_without_selected(array, &selected(), short_lines),
        SelectedAction::SetNull => LineStringArray::from_parts(
            rows_nulled_where_selected(&array.rows, &[], &array.coords, selection),
            array.coords.clone(),
        ),
    }
}

pub fn multi_points_with_selected(
    array: &MultiPointArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
    short_lines: ShortLines,
) -> MultiPointArray {
    let selected = || selected_coords_bitmap(&array.coords, selection);
    match action {
        SelectedAction::SetEmpty => {
            MultiPointArray::from_parts(array.rows.clone(), emptied(&array.coords, &selected()))
        }
        SelectedAction::Drop => multi_points_without_selected(array, &selected(), short_lines),
        SelectedAction::SetNull => MultiPointArray::from_parts(
            rows_nulled_where_selected(&array.rows, &[], &array.coords, selection),
            array.coords.clone(),
        ),
    }
}

pub fn polygons_with_selected(
    array: &PolygonArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
    short_lines: ShortLines,
) -> PolygonArray {
    let selected = || selected_coords_bitmap(&array.coords, selection);
    match action {
        SelectedAction::SetEmpty => PolygonArray::from_parts(
            array.rows.clone(),
            array.rings.clone(),
            emptied(&array.coords, &selected()),
        ),
        SelectedAction::Drop => polygons_without_selected(array, &selected(), short_lines, true),
        SelectedAction::SetNull => PolygonArray::from_parts(
            rows_nulled_where_selected(&array.rows, &[&array.rings], &array.coords, selection),
            array.rings.clone(),
            array.coords.clone(),
        ),
    }
}

pub fn multi_line_strings_with_selected(
    array: &MultiLineStringArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
    short_lines: ShortLines,
) -> MultiLineStringArray {
    let selected = || selected_coords_bitmap(&array.coords, selection);
    match action {
        SelectedAction::SetEmpty => MultiLineStringArray::from_parts(
            array.rows.clone(),
            array.lines.clone(),
            emptied(&array.coords, &selected()),
        ),
        SelectedAction::Drop => {
            multi_line_strings_without_selected(array, &selected(), short_lines)
        }
        SelectedAction::SetNull => MultiLineStringArray::from_parts(
            rows_nulled_where_selected(&array.rows, &[&array.lines], &array.coords, selection),
            array.lines.clone(),
            array.coords.clone(),
        ),
    }
}

pub fn multi_polygons_with_selected(
    array: &MultiPolygonArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
    short_lines: ShortLines,
) -> MultiPolygonArray {
    let selected = || selected_coords_bitmap(&array.coords, selection);
    match action {
        SelectedAction::SetEmpty => MultiPolygonArray::from_parts(
            array.rows.clone(),
            array.polygons.clone(),
            array.rings.clone(),
            emptied(&array.coords, &selected()),
        ),
        SelectedAction::Drop => {
            multi_polygons_without_selected(array, &selected(), short_lines, true)
        }
        SelectedAction::SetNull => MultiPolygonArray::from_parts(
            rows_nulled_where_selected(
                &array.rows,
                &[&array.polygons, &array.rings],
                &array.coords,
                selection,
            ),
            array.polygons.clone(),
            array.rings.clone(),
            array.coords.clone(),
        ),
    }
}
