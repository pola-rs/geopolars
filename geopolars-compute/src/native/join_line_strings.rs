//! Joining the linestrings of each row into one, end to start.
//! See `join_line_strings` in Python for the rules.
//!
//! The parts of a row already lie one after the other in the coordinates,
//! so a join only needs new offsets, and the coordinates are shared.
//! Only `drop_repeated_joins` copies them, as in [super::selected_coords](mod@super::selected_coords).
//!
//! # Const generic flags
//!
//! [for_each_join] passes `TOLERANCE_COINCIDENCE`, `UNIQUE_Z` and `UNIQUE_M` on to [coincide].
//! See [super::has_repeats_in_succession](mod@super::has_repeats_in_succession) for them.

use std::iter::Peekable;

use geopolars_core::{LineStringArray, List, MultiLineStringArray, Tolerance, ValidList};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::Offsets;

use super::has_repeats_in_succession::{coincide, unique_axes};
use super::selected_coords::{line_strings_without_selected, multi_line_strings_without_selected};
use super::too_short::TooShort;

/// What a joined linestring does at a gap, a join that does not coincide.
/// See `gaps` of `join_line_strings` in Python.
/// Its `"split"` is [line_strings_joined_where_coinciding].
#[derive(Debug, Clone, Copy)]
pub enum Gaps {
    Null,
    Bridge,
}

fn for_each_join_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &MultiLineStringArray,
    tolerance: Tolerance,
    mut on_join: impl FnMut(usize, usize, bool),
) {
    let rows = array.rows.ignoring_validity();
    for row in 0..rows.len() {
        let mut last_end = None;
        for line in rows.range(row) {
            let vertices = array.lines.range(line);
            if vertices.is_empty() {
                continue;
            }
            if let Some(last_end) = last_end {
                let coincides = coincide::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
                    &array.coords,
                    last_end,
                    vertices.start,
                    tolerance,
                );
                on_join(row, vertices.start, coincides);
            }
            last_end = Some(vertices.end - 1);
        }
    }
}

/// Calls `on_join(row, start, coincides)` for every join, in order.
/// A join is the last vertex of a non-empty part followed by the first vertex `start` of the next non-empty part.
/// Empty parts have no vertices to join, and are skipped.
fn for_each_join(
    array: &MultiLineStringArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
    on_join: impl FnMut(usize, usize, bool),
) {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        TOLERANCE_COINCIDENCE = tolerance.is_some(), UNIQUE_Z = unique_z, UNIQUE_M = unique_m;
        for_each_join_with::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            array,
            tolerance.unwrap_or_default(),
            on_join,
        )
    )
}

/// The vertices of all parts of each row, as one list per row.
fn vertices_per_row(array: &MultiLineStringArray) -> ValidList {
    let lines = array.lines.offsets().buffer();
    let offsets = array.rows.ignoring_validity().offsets().buffer();
    let offsets = offsets.iter().map(|&row| lines[row as usize]).collect();
    // SAFETY: the offsets of the lines grow, and so do those of the rows that index them.
    ValidList::from_offsets(unsafe { Offsets::new_unchecked(offsets) }.into())
}

/// One linestring per row of `array`, out of all its parts.
/// A row with a gap is null with [Gaps::Null].
/// With `drop_repeated_joins=true`, the first vertex of a part is dropped where it coincides with the end of the one before.
pub fn joined_line_strings(
    array: &MultiLineStringArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
    gaps: Gaps,
    drop_repeated_joins: bool,
) -> LineStringArray {
    let rows = vertices_per_row(array);
    let null_gaps = matches!(gaps, Gaps::Null);
    if !null_gaps && !drop_repeated_joins {
        let rows = List::from_offsets(rows.offsets().clone(), array.rows.validity().cloned());
        return LineStringArray::from_parts(rows, array.coords.clone());
    }

    let mut gapless = null_gaps.then(|| MutableBitmap::from_len_set(array.rows.len()));
    let mut dropped =
        drop_repeated_joins.then(|| MutableBitmap::from_len_zeroed(array.coords.len()));
    for_each_join(
        array,
        tolerance,
        unique_z,
        unique_m,
        |row, start, coincides| match (coincides, &mut gapless, &mut dropped) {
            (false, Some(gapless), _) => gapless.set(row, false),
            (true, _, Some(dropped)) => dropped.set(start, true),
            _ => {}
        },
    );
    let gapless = gapless.map(Bitmap::from);
    let validity = combine_validities_and(array.rows.validity(), gapless.as_ref());
    let joined = LineStringArray::from_parts(
        List::from_offsets(rows.offsets().clone(), validity),
        array.coords.clone(),
    );
    match dropped {
        Some(dropped) => line_strings_without_selected(&joined, &dropped.freeze(), TooShort::Keep),
        None => joined,
    }
}

/// The offsets of the lines `array` is split into at the vertices `gaps` gives, as `(row, start)` in order.
/// Each row with vertices gets one line more than it has gaps, and one without gets none.
fn lines_split_at(
    array: &MultiLineStringArray,
    rows: &ValidList,
    mut gaps: Peekable<impl Iterator<Item = (usize, usize)>>,
) -> (List, ValidList) {
    let mut row_offsets = Offsets::<i64>::with_capacity(rows.len());
    let mut line_offsets = Offsets::<i64>::with_capacity(array.lines.len());
    const NO_OVERFLOW: &str = "no more lines than vertices";
    for (row, vertices) in rows.ranges().enumerate() {
        let mut line_start = vertices.start;
        let mut lines = 0;
        while let Some((_, start)) = gaps.next_if(|&(gap_row, _)| gap_row == row) {
            line_offsets
                .try_push(start - line_start)
                .expect(NO_OVERFLOW);
            line_start = start;
            lines += 1;
        }
        if !vertices.is_empty() {
            line_offsets
                .try_push(vertices.end - line_start)
                .expect(NO_OVERFLOW);
            lines += 1;
        }
        row_offsets.try_push(lines).expect(NO_OVERFLOW);
    }
    (
        List::from_offsets(row_offsets.into(), array.rows.validity().cloned()),
        ValidList::from_offsets(line_offsets.into()),
    )
}

/// `array` with the parts of each row joined where they coincide, and split at each gap.
/// See [joined_line_strings] for `drop_repeated_joins`.
pub fn line_strings_joined_where_coinciding(
    array: &MultiLineStringArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
    drop_repeated_joins: bool,
) -> MultiLineStringArray {
    let mut gaps = Vec::new();
    let mut dropped =
        drop_repeated_joins.then(|| MutableBitmap::from_len_zeroed(array.coords.len()));
    for_each_join(
        array,
        tolerance,
        unique_z,
        unique_m,
        |row, start, coincides| match (coincides, &mut dropped) {
            (false, _) => gaps.push((row, start)),
            (true, Some(dropped)) => dropped.set(start, true),
            (true, None) => {}
        },
    );
    let (rows, lines) =
        lines_split_at(array, &vertices_per_row(array), gaps.into_iter().peekable());
    let joined = MultiLineStringArray::from_parts(rows, lines, array.coords.clone());
    match dropped {
        Some(dropped) => {
            multi_line_strings_without_selected(&joined, &dropped.freeze(), TooShort::Keep)
        }
        None => joined,
    }
}
