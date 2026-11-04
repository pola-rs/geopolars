//! Joining the linestrings of each row into one, end to start.
//! See `join_line_strings` in Python for the rules.
//!
//! The parts of a row already lie one after the other in the coordinates,
//! so a join only needs new offsets, and the coordinates are shared.
//! Only `drop_repeated_joins` copies them, through [super::selected_coords](mod@super::selected_coords).

use std::iter::Peekable;

use geopolars_core::{LineStringArray, List, MultiLineStringArray, ValidList};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::Offsets;

use super::has_repeats_in_succession::{with_coincide, Coincidence};
use super::selected_coords::{line_strings_without_selected, multi_line_strings_without_selected};
use super::too_short::TooShort;

/// What [multi_line_strings_joined] does at a gap: a join whose two vertices do not coincide.
/// See `gaps` of `join_line_strings` in Python.
/// `gaps="split"` gives multilinestrings, so it has no variant here.
/// See [multi_line_strings_split_at_gaps] for it.
#[derive(Debug, Clone, Copy)]
pub enum Gaps {
    Null,
    Bridge,
}

fn for_each_join_with(
    array: &MultiLineStringArray,
    coincide: impl Fn(usize, usize) -> bool,
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
                on_join(row, vertices.start, coincide(last_end, vertices.start));
            }
            last_end = Some(vertices.end - 1);
        }
    }
}

/// Calls `on_join(row, start, coincides)` for every join
fn for_each_join(
    array: &MultiLineStringArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    on_join: impl FnMut(usize, usize, bool),
) {
    with_coincide!(&array.coords, coincidence, unique_z, unique_m; |coincide| {
        for_each_join_with(array, coincide, on_join)
    })
}

fn vertices_per_row(array: &MultiLineStringArray) -> ValidList {
    let lines = array.lines.offsets().buffer();
    let offsets = array.rows.ignoring_validity().offsets().buffer();
    let offsets = offsets.iter().map(|&row| lines[row as usize]).collect();
    // SAFETY: the row offsets never decrease, and neither do the line offsets they index,
    // so the line offset at each row offset never decreases either.
    ValidList::from_offsets(unsafe { Offsets::new_unchecked(offsets) }.into())
}

/// One linestring per row of `array`, out of all its parts.
/// A row with a gap is null with [Gaps::Null].
/// With `drop_repeated_joins=true`, the first vertex of a part is dropped where it coincides with the end of the one before.
pub fn multi_line_strings_joined(
    array: &MultiLineStringArray,
    coincidence: Coincidence,
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
        coincidence,
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
    // Only the first vertex of a later part is dropped, so a line keeps every vertex of its first part.
    // It is only too short if that part already was.
    match dropped {
        Some(dropped) => line_strings_without_selected(&joined, &dropped.freeze(), TooShort::Keep),
        None => joined,
    }
}

/// The row and line offsets of `rows`, split into a new line at each gap.
/// `gaps` holds `(row, first vertex after the gap)`.
/// A row with vertices gets one line more than it has gaps. An empty row gets none.
fn lines_split_at(
    array: &MultiLineStringArray,
    rows: &ValidList,
    mut gaps: Peekable<impl Iterator<Item = (usize, usize)>>,
) -> (List, ValidList) {
    let mut row_offsets = Offsets::<i64>::with_capacity(rows.len());
    let mut line_offsets = Offsets::<i64>::with_capacity(array.lines.len());
    const NO_OVERFLOW: &str = "no more lines or vertices than the input's offsets index";
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

/// [multi_line_strings_joined], but starting a new part at each gap.
pub fn multi_line_strings_split_at_gaps(
    array: &MultiLineStringArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    drop_repeated_joins: bool,
) -> MultiLineStringArray {
    let mut gaps = Vec::new();
    let mut dropped =
        drop_repeated_joins.then(|| MutableBitmap::from_len_zeroed(array.coords.len()));
    for_each_join(
        array,
        coincidence,
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
    // See `multi_line_strings_joined` on `TooShort::Keep`. A split line starts at a gap, which is never dropped.
    match dropped {
        Some(dropped) => {
            multi_line_strings_without_selected(&joined, &dropped.freeze(), TooShort::Keep)
        }
        None => joined,
    }
}
