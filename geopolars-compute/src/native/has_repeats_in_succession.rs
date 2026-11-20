//! Whether a geometry has two points in a row that coincide.
//! Implements the Python `has_repeats_in_succession`.
//!
//! [repeats_in_succession] marks the repeated vertices, in one pass.
//! The kernels here fold that mask up to the rows.
//! [super::remove_repeats_in_succession](mod@super::remove_repeats_in_succession) drops them.
//!
//! # Const generic flags
//!
//! [coincide_exactly], [coincide_within_abs], and [coincide_within_rel] compare z only with `Z=true`,
//! and m only with `M=true`.
//! `with_coincide!` first leaves out the axes the Python `include_z` and `include_m` exclude
//! (see [ValidCoords::with_only_axes]).
//! It then sets `Z` and `M` to the axes that are left.

use geopolars_views::{
    is_close, is_close_abs, same_place, same_place_within, same_place_within_abs, LineStringArray,
    List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray, PolygonArray,
    Tolerance, ValidCoords, ValidList,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::datatypes::ArrowDataType;

use crate::passing::valid_rows_passing;

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
    coords.m().expect("an m, with `M`")
}

/// See [Coincidence::Exact].
#[inline(always)]
pub(super) fn coincide_exactly<const Z: bool, const M: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
) -> bool {
    same_place::<Z>(coords, i, j)
        && (!M || {
            let m = m_values(coords);
            m[i] == m[j]
        })
}

#[inline(always)]
pub(super) fn coincide_within_abs<const Z: bool, const M: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
    abs_tol: f64,
) -> bool {
    same_place_within_abs::<Z>(coords, i, j, abs_tol)
        && (!M || {
            let m = m_values(coords);
            is_close_abs(m[i], m[j], abs_tol)
        })
}

#[inline(always)]
pub(super) fn coincide_within_rel<const Z: bool, const M: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
    tolerance: Tolerance,
) -> bool {
    same_place_within::<Z>(coords, i, j, tolerance)
        && (!M || {
            let m = m_values(coords);
            is_close(m[i], m[j], tolerance)
        })
}

/// Evaluates `$body` with `$coincide` bound to a closure `|i, j| -> bool`:
/// whether coordinates `i` and `j` of `$coords` coincide by `$coincidence`.
/// Only the axes of `$coords` that `$include_z` and `$include_m` keep are compared.
/// `$body` is expanded once per variant of [Coincidence],
/// so a pass that takes the closure has no per-vertex match on the variant.
macro_rules! with_coincide {
    (
        $coords:expr, $coincidence:expr, $include_z:expr, $include_m:expr;
        |$coincide:ident| $body:expr
    ) => {{
        use $crate::native::has_repeats_in_succession::{
            coincide_exactly, Coincidence, coincide_within_abs, coincide_within_rel,
        };
        let compared: &geopolars_views::ValidCoords =
            &$coords.with_only_axes($include_z, $include_m);
        $crate::with_const_flags!(
            Z = compared.z().is_some(), M = compared.m().is_some();
            match $coincidence {
                Coincidence::Exact => {
                    let $coincide =
                        move |i, j| coincide_exactly::<Z, M>(compared, i, j);
                    $body
                }
                Coincidence::WithinAbs(abs_tol) => {
                    let $coincide = move |i, j| {
                        coincide_within_abs::<Z, M>(compared, i, j, abs_tol)
                    };
                    $body
                }
                Coincidence::WithinRel(tolerance) => {
                    let $coincide = move |i, j| {
                        coincide_within_rel::<Z, M>(compared, i, j, tolerance)
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
    include_z: bool,
    include_m: bool,
) -> MutableBitmap {
    with_coincide!(coords, coincidence, include_z, include_m; |coincide| {
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

pub fn points_have_repeats_in_succession(array: &PointArray) -> BooleanArray {
    let coords = &array.coords;
    let values = Bitmap::new_zeroed(coords.ignoring_validity().len());
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

pub fn multi_points_have_repeats_in_succession(
    array: &MultiPointArray,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
) -> BooleanArray {
    let rows = array.rows.ignoring_validity();
    let repeated = repeats_in_succession(rows, &array.coords, coincidence, include_z, include_m);
    rows_with_repeats(&array.rows, &[], repeated)
}

pub fn line_strings_have_repeats_in_succession(
    array: &LineStringArray,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
) -> BooleanArray {
    let rows = array.rows.ignoring_validity();
    let repeated = repeats_in_succession(rows, &array.coords, coincidence, include_z, include_m);
    rows_with_repeats(&array.rows, &[], repeated)
}

pub fn multi_line_strings_have_repeats_in_succession(
    array: &MultiLineStringArray,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
) -> BooleanArray {
    let repeated = repeats_in_succession(
        &array.lines,
        &array.coords,
        coincidence,
        include_z,
        include_m,
    );
    rows_with_repeats(&array.rows, &[&array.lines], repeated)
}

pub fn polygons_have_repeats_in_succession(
    array: &PolygonArray,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
) -> BooleanArray {
    let repeated = repeats_in_succession(
        &array.rings,
        &array.coords,
        coincidence,
        include_z,
        include_m,
    );
    rows_with_repeats(&array.rows, &[&array.rings], repeated)
}

pub fn multi_polygons_have_repeats_in_succession(
    array: &MultiPolygonArray,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
) -> BooleanArray {
    let repeated = repeats_in_succession(
        &array.rings,
        &array.coords,
        coincidence,
        include_z,
        include_m,
    );
    rows_with_repeats(&array.rows, &[&array.polygons, &array.rings], repeated)
}
