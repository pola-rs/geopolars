//! Whether a geometry has two points in a row that coincide.
//! See `has_repeats_in_succession` in Python for the rules.
//!
//! [repeats_in_succession] marks the repeated vertices, in one pass.
//! The kernels here fold that mask up to the rows.
//! [super::remove_repeats_in_succession](mod@super::remove_repeats_in_succession) drops them.
//!
//! # Const generic flags
//!
//! ## `TOLERANCE_COINCIDENCE`
//!
//! [coincide] and the pass are generic over `TOLERANCE_COINCIDENCE`.
//! If `TOLERANCE_COINCIDENCE=true`, two points coincide when they lie at the same place within `tolerance`
//! ([same_place_within]).
//! If `TOLERANCE_COINCIDENCE=false`, they have to lie at exactly the same place ([same_place]).
//!
//! ## `UNIQUE_Z`
//!
//! [coincide] and the pass are generic over `UNIQUE_Z`.
//! If `UNIQUE_Z=true`, two points only coincide when their z values do too.
//! A runtime `unique_z` is ignored for coordinates without a z.
//!
//! ## `UNIQUE_M`
//!
//! [coincide] and the pass are generic over `UNIQUE_M`.
//! If `UNIQUE_M=true`, two points only coincide when their m values do too.
//! A runtime `unique_m` is ignored for coordinates without an m.

use geopolars_core::{
    is_close, same_place, same_place_within, LineStringArray, List, MultiLineStringArray,
    MultiPointArray, MultiPolygonArray, PointArray, PolygonArray, Tolerance, ValidCoords,
    ValidList,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::datatypes::ArrowDataType;

use crate::passing::valid_rows_passing;

#[inline(always)]
pub(super) fn coincide<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
    tolerance: Tolerance,
) -> bool {
    let m_coincides = || {
        let m = coords.m().expect("an m, with `UNIQUE_M`");
        if TOLERANCE_COINCIDENCE {
            is_close(m[i], m[j], tolerance)
        } else {
            m[i] == m[j]
        }
    };
    let place_coincides = if TOLERANCE_COINCIDENCE {
        same_place_within::<UNIQUE_Z>(coords, i, j, tolerance)
    } else {
        same_place::<UNIQUE_Z>(coords, i, j)
    };
    place_coincides && (!UNIQUE_M || m_coincides())
}

/// `unique_z` and `unique_m`, only for an axis `coords` has.
fn unique_axes(coords: &ValidCoords, unique_z: bool, unique_m: bool) -> (bool, bool) {
    let dim = coords.dim();
    (unique_z && dim.has_z(), unique_m && dim.has_m())
}

// TODO: Not yet a SIMD pass, but is a minor candidate for it.
// Without a tolerance, coinciding is transitive.
// A vertex then coincides with the last unset one before it exactly when it coincides with the one right before it.
// So the mask could be built by `bitmap_from_fn` from `coincide(i - 1, i)`, as `selected_coords` is.
// Afterwards, only the first vertex of every list needs to be unset.
fn repeats_in_succession_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    lists: &ValidList,
    coords: &ValidCoords,
    tolerance: Tolerance,
) -> MutableBitmap {
    let mut repeated = MutableBitmap::from_len_zeroed(coords.len());
    for list in lists.ranges() {
        let mut last_kept = list.start;
        for i in list.start + 1..list.end {
            if coincide::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
                coords, last_kept, i, tolerance,
            ) {
                repeated.set(i, true);
            } else {
                last_kept = i;
            }
        }
    }
    repeated
}

/// Which of `coords` repeat a vertex, as set bits.
/// A vertex is set when it [coincides](coincide) with the last unset vertex before it in its list of `lists`.
/// The first vertex of a list is never set.
///
/// A list holds a set vertex exactly when two vertices in a row coincide.
/// The first set vertex was compared with the one right before it.
/// And if two vertices in a row coincide, the second is set, unless an earlier one already is.
pub(super) fn repeats_in_succession(
    lists: &ValidList,
    coords: &ValidCoords,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> MutableBitmap {
    let (unique_z, unique_m) = unique_axes(coords, unique_z, unique_m);
    with_const_flags!(
        TOLERANCE_COINCIDENCE = tolerance.is_some(), UNIQUE_Z = unique_z, UNIQUE_M = unique_m;
        repeats_in_succession_with::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            lists,
            coords,
            tolerance.unwrap_or_default(),
        )
    )
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
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let rows = array.rows.ignoring_validity();
    let repeated = repeats_in_succession(rows, &array.coords, tolerance, unique_z, unique_m);
    rows_with_repeats(&array.rows, &[], repeated)
}

pub fn line_strings_have_repeats_in_succession(
    array: &LineStringArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let rows = array.rows.ignoring_validity();
    let repeated = repeats_in_succession(rows, &array.coords, tolerance, unique_z, unique_m);
    rows_with_repeats(&array.rows, &[], repeated)
}

pub fn multi_line_strings_have_repeats_in_succession(
    array: &MultiLineStringArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let repeated =
        repeats_in_succession(&array.lines, &array.coords, tolerance, unique_z, unique_m);
    rows_with_repeats(&array.rows, &[&array.lines], repeated)
}

pub fn polygons_have_repeats_in_succession(
    array: &PolygonArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let repeated =
        repeats_in_succession(&array.rings, &array.coords, tolerance, unique_z, unique_m);
    rows_with_repeats(&array.rows, &[&array.rings], repeated)
}

pub fn multi_polygons_have_repeats_in_succession(
    array: &MultiPolygonArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let repeated =
        repeats_in_succession(&array.rings, &array.coords, tolerance, unique_z, unique_m);
    rows_with_repeats(&array.rows, &[&array.polygons, &array.rings], repeated)
}
