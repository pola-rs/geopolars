//! Whether a geometry has two points in a row that coincide.
//! See `has_repeats_in_succession` in Python for the rules.
//!
//! [repeats_in_succession] marks the repeated vertices, in one pass.
//! The kernels here fold that mask up to the rows.
//! [super::remove_repeats_in_succession](mod@super::remove_repeats_in_succession) drops them.
//!
//! # Const generic flags
//!
//! [coincide_exactly], [coincide_within_abs], and [coincide_within_rel] are generic over `UNIQUE_Z` and `UNIQUE_M`.
//!
//! ## `UNIQUE_Z`
//!
//! If `UNIQUE_Z=true`, two points only coincide when their z values do too.
//! A runtime `unique_z` is ignored for coordinates without a z.
//!
//! ## `UNIQUE_M`
//!
//! If `UNIQUE_M=true`, two points only coincide when their m values do too.
//! A runtime `unique_m` is ignored for coordinates without an m.

use geopolars_core::{
    is_close, is_close_abs, same_place, same_place_within, same_place_within_abs, LineStringArray,
    List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray, PolygonArray,
    Tolerance, ValidCoords, ValidList,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::datatypes::ArrowDataType;

use crate::passing::valid_rows_passing;

/// When two points coincide, given the `abs_tol` and `rel_tol` of an expression.
/// Build it with [Coincidence::new].
/// Each variant relies on the tolerance range in its doc.
#[derive(Debug, Clone, Copy)]
pub enum Coincidence {
    /// At exactly the same place.
    Exact,
    /// [is_close_abs] on every axis, with a finite, positive `abs_tol`.
    WithinAbs(f64),
    /// [is_close] on every axis, with a positive `rel_tol` or an infinite `abs_tol`.
    /// An infinite `abs_tol` is not [Coincidence::WithinAbs], see [is_close_abs].
    WithinRel(Tolerance),
}

impl Coincidence {
    /// A tolerance that is not given counts as `0.0`.
    /// Neither tolerance may be negative or NaN.
    pub fn new(abs_tol: Option<f64>, rel_tol: Option<f64>) -> Self {
        let abs_tol = abs_tol.unwrap_or(0.0);
        let rel_tol = rel_tol.unwrap_or(0.0);
        debug_assert!(abs_tol >= 0.0, "abs_tol={abs_tol}");
        debug_assert!(rel_tol >= 0.0, "rel_tol={rel_tol}");
        if rel_tol > 0.0 || abs_tol.is_infinite() {
            Self::WithinRel(Tolerance { abs_tol, rel_tol })
        } else if abs_tol > 0.0 {
            Self::WithinAbs(abs_tol)
        } else {
            Self::Exact
        }
    }
}

fn m_values(coords: &ValidCoords) -> &[f64] {
    coords.m().expect("an m, with `UNIQUE_M`")
}

/// See [Coincidence::Exact].
#[inline(always)]
pub(super) fn coincide_exactly<const UNIQUE_Z: bool, const UNIQUE_M: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
) -> bool {
    same_place::<UNIQUE_Z>(coords, i, j)
        && (!UNIQUE_M || {
            let m = m_values(coords);
            m[i] == m[j]
        })
}

/// See [Coincidence::WithinAbs].
#[inline(always)]
pub(super) fn coincide_within_abs<const UNIQUE_Z: bool, const UNIQUE_M: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
    abs_tol: f64,
) -> bool {
    same_place_within_abs::<UNIQUE_Z>(coords, i, j, abs_tol)
        && (!UNIQUE_M || {
            let m = m_values(coords);
            is_close_abs(m[i], m[j], abs_tol)
        })
}

/// See [Coincidence::WithinRel].
#[inline(always)]
pub(super) fn coincide_within_rel<const UNIQUE_Z: bool, const UNIQUE_M: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
    tolerance: Tolerance,
) -> bool {
    same_place_within::<UNIQUE_Z>(coords, i, j, tolerance)
        && (!UNIQUE_M || {
            let m = m_values(coords);
            is_close(m[i], m[j], tolerance)
        })
}

/// `unique_z` and `unique_m`, only for an axis `coords` has.
pub(super) fn unique_axes(coords: &ValidCoords, unique_z: bool, unique_m: bool) -> (bool, bool) {
    let dim = coords.dim();
    (unique_z && dim.has_z(), unique_m && dim.has_m())
}

/// Evaluates `$body` with `$coincide` bound to a closure `|i, j| -> bool`:
/// whether coordinates `i` and `j` of `$coords` coincide by `$coincidence`.
/// `$unique_z` and `$unique_m` only count for an axis `$coords` has ([unique_axes]).
/// `$body` is expanded once per variant of [Coincidence],
/// so a pass that takes the closure has no per-vertex match on the variant.
macro_rules! with_coincide {
    (
        $coords:expr, $coincidence:expr, $unique_z:expr, $unique_m:expr;
        |$coincide:ident| $body:expr
    ) => {{
        use $crate::native::has_repeats_in_succession::{
            coincide_exactly, Coincidence, coincide_within_abs, coincide_within_rel, unique_axes,
        };
        let coords: &geopolars_core::ValidCoords = $coords;
        let (unique_z, unique_m) = unique_axes(coords, $unique_z, $unique_m);
        $crate::with_const_flags!(
            UNIQUE_Z = unique_z, UNIQUE_M = unique_m;
            match $coincidence {
                Coincidence::Exact => {
                    let $coincide =
                        move |i, j| coincide_exactly::<UNIQUE_Z, UNIQUE_M>(coords, i, j);
                    $body
                }
                Coincidence::WithinAbs(abs_tol) => {
                    let $coincide = move |i, j| {
                        coincide_within_abs::<UNIQUE_Z, UNIQUE_M>(coords, i, j, abs_tol)
                    };
                    $body
                }
                Coincidence::WithinRel(tolerance) => {
                    let $coincide = move |i, j| {
                        coincide_within_rel::<UNIQUE_Z, UNIQUE_M>(coords, i, j, tolerance)
                    };
                    $body
                }
            }
        )
    }};
}
pub(super) use with_coincide;

// TODO: Not yet a SIMD pass, but is a minor candidate for it.
// With `Coincidence::Exact`, coinciding is transitive.
// A vertex then coincides with the last unset one before it exactly when it coincides with the one right before it.
// So the mask could be built by `bitmap_from_fn` from `coincide_exactly(i - 1, i)`, as `selected_coords` is.
// Afterwards, only the first vertex of every list needs to be unset.
fn repeats_in_succession_with(
    lists: &ValidList,
    vertex_count: usize,
    coincide: impl Fn(usize, usize) -> bool,
) -> MutableBitmap {
    let mut repeated = MutableBitmap::from_len_zeroed(vertex_count);
    for list in lists.ranges() {
        let mut last_kept = list.start;
        for i in list.start + 1..list.end {
            if coincide(last_kept, i) {
                repeated.set(i, true);
            } else {
                last_kept = i;
            }
        }
    }
    repeated
}

/// Which of `coords` repeat a vertex, as set bits.
/// A vertex is set when it [coincides](Coincidence) with the last unset vertex before it in its list of `lists`.
/// The first vertex of a list is never set.
///
/// A list holds a set vertex exactly when two vertices in a row coincide.
/// The first set vertex was compared with the one right before it.
/// And if two vertices in a row coincide, the second is set, unless an earlier one already is.
pub(super) fn repeats_in_succession(
    lists: &ValidList,
    coords: &ValidCoords,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
) -> MutableBitmap {
    with_coincide!(coords, coincidence, unique_z, unique_m; |coincide| {
        repeats_in_succession_with(lists, coords.len(), coincide)
    })
}

/// A `bool` per row of `rows`, `true` where the row holds a vertex set in `repeated`.
/// `inner_lists` are the list levels below the rows, outermost first.
fn rows_with_repeats(
    rows: &List,
    inner_lists: &[&ValidList],
    repeated: MutableBitmap,
) -> BooleanArray {
    let not_repeated = !&Bitmap::from(repeated);
    let passing = valid_rows_passing(rows, inner_lists, Some(not_repeated));
    let values = passing.map_or_else(|| Bitmap::new_zeroed(rows.len()), |passing| !&passing);
    BooleanArray::new(ArrowDataType::Boolean, values, rows.validity().cloned())
}

/// A point has nothing to coincide with, so this is `false` for every point that is there.
pub fn points_have_repeats_in_succession(array: &PointArray) -> BooleanArray {
    let coords = &array.coords;
    let values = Bitmap::new_zeroed(coords.ignoring_validity().len());
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

pub fn multi_points_have_repeats_in_succession(
    array: &MultiPointArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let rows = array.rows.ignoring_validity();
    let repeated = repeats_in_succession(rows, &array.coords, coincidence, unique_z, unique_m);
    rows_with_repeats(&array.rows, &[], repeated)
}

pub fn line_strings_have_repeats_in_succession(
    array: &LineStringArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let rows = array.rows.ignoring_validity();
    let repeated = repeats_in_succession(rows, &array.coords, coincidence, unique_z, unique_m);
    rows_with_repeats(&array.rows, &[], repeated)
}

pub fn multi_line_strings_have_repeats_in_succession(
    array: &MultiLineStringArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let repeated =
        repeats_in_succession(&array.lines, &array.coords, coincidence, unique_z, unique_m);
    rows_with_repeats(&array.rows, &[&array.lines], repeated)
}

pub fn polygons_have_repeats_in_succession(
    array: &PolygonArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let repeated =
        repeats_in_succession(&array.rings, &array.coords, coincidence, unique_z, unique_m);
    rows_with_repeats(&array.rows, &[&array.rings], repeated)
}

pub fn multi_polygons_have_repeats_in_succession(
    array: &MultiPolygonArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let repeated =
        repeats_in_succession(&array.rings, &array.coords, coincidence, unique_z, unique_m);
    rows_with_repeats(&array.rows, &[&array.polygons, &array.rings], repeated)
}
