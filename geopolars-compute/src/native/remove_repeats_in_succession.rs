//! Removing the points that coincide with the one kept before them.
//! Implements the Python `remove_repeats_in_succession`.
//!
//! Every kernel marks the repeated vertices ([repeats_in_succession]).
//! Linestrings and rings then keep their last vertex ([keep_last_vertices]).
//! The marked vertices are dropped as in [super::selected_coords](mod@super::selected_coords),
//! which also handles the linestrings and rings left too short.
//! As rings keep both ends, the drop does not check whether they are still closed.

use std::ops::Range;

use geopolars_views::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
    ValidList,
};
use polars_arrow::bitmap::MutableBitmap;

use super::has_repeats_in_succession::{repeats_in_succession, Coincidence};
use super::selected_coords::{
    line_strings_without_selected, multi_line_strings_without_selected,
    multi_points_without_selected, multi_polygons_without_selected, polygons_without_selected,
};
use super::too_short::TooShort;

/// The `collapsed_lines` of the Python `remove_repeats_in_succession`.
#[derive(Debug, Clone, Copy)]
pub enum CollapsedLines {
    KeepEnds,
    SingleVertex,
}

impl CollapsedLines {
    fn keeps_both_ends(self) -> bool {
        matches!(self, CollapsedLines::KeepEnds)
    }
}

/// The `collapsed_lines` and `too_short` of the Python `remove_repeats_in_succession`.
#[derive(Debug, Clone, Copy)]
pub struct LineOptions {
    pub collapsed_lines: CollapsedLines,
    pub too_short: TooShort,
}

/// Unsets the last vertex of each of `lists` in `repeated`, so that it is kept.
/// The last vertex kept before it is set instead, unless that is the first vertex.
/// A list that collapsed into its first vertex keeps its last one only if `keeps_both_ends=true`.
fn keep_last_vertices(lists: &ValidList, repeated: &mut MutableBitmap, keeps_both_ends: bool) {
    for Range { start, end } in lists.ranges() {
        if end - start < 2 || !repeated.get(end - 1) {
            continue;
        }
        let last_kept = (start..end - 1)
            .rev()
            .find(|&i| !repeated.get(i))
            .expect("the first vertex, which is never repeated");
        if last_kept != start {
            repeated.set(last_kept, true);
            repeated.set(end - 1, false);
        } else if keeps_both_ends {
            repeated.set(end - 1, false);
        }
    }
}

pub fn line_strings_without_repeats_in_succession(
    array: &LineStringArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    options: LineOptions,
) -> LineStringArray {
    let rows = array.rows.ignoring_validity();
    let mut repeated = repeats_in_succession(rows, &array.coords, coincidence, unique_z, unique_m);
    keep_last_vertices(
        rows,
        &mut repeated,
        options.collapsed_lines.keeps_both_ends(),
    );
    line_strings_without_selected(array, &repeated.freeze(), options.too_short)
}

/// Takes no [LineOptions],
/// as the points of a multipoint have no ends to keep, and are never too short.
pub fn multi_points_without_repeats_in_succession(
    array: &MultiPointArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
) -> MultiPointArray {
    let rows = array.rows.ignoring_validity();
    let repeated = repeats_in_succession(rows, &array.coords, coincidence, unique_z, unique_m);
    multi_points_without_selected(array, &repeated.freeze(), TooShort::Keep)
}

/// Takes no [CollapsedLines], as rings always keep both ends.
pub fn polygons_without_repeats_in_succession(
    array: &PolygonArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    too_short: TooShort,
) -> PolygonArray {
    let rings = &array.rings;
    let mut repeated = repeats_in_succession(rings, &array.coords, coincidence, unique_z, unique_m);
    keep_last_vertices(rings, &mut repeated, true);
    polygons_without_selected(array, &repeated.freeze(), too_short, false)
}

pub fn multi_line_strings_without_repeats_in_succession(
    array: &MultiLineStringArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    options: LineOptions,
) -> MultiLineStringArray {
    let lines = &array.lines;
    let mut repeated = repeats_in_succession(lines, &array.coords, coincidence, unique_z, unique_m);
    keep_last_vertices(
        lines,
        &mut repeated,
        options.collapsed_lines.keeps_both_ends(),
    );
    multi_line_strings_without_selected(array, &repeated.freeze(), options.too_short)
}

/// Takes no [CollapsedLines], as rings always keep both ends.
pub fn multi_polygons_without_repeats_in_succession(
    array: &MultiPolygonArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    too_short: TooShort,
) -> MultiPolygonArray {
    let rings = &array.rings;
    let mut repeated = repeats_in_succession(rings, &array.coords, coincidence, unique_z, unique_m);
    keep_last_vertices(rings, &mut repeated, true);
    multi_polygons_without_selected(array, &repeated.freeze(), too_short, false)
}
