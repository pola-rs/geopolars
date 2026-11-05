//! Whether two geometries are almost coincident/the same.
//! See `is_equal` in Python for the rules.
//!
//! Every kernel takes `a`, and `b` of the same length (row by row) or of a single row (against every row of `a`).
//! Every comparison stops at the first part length or value that does not match.
//!
//! Every kernel takes a [Coincidence], matched once in `are_close!`.
//! The `*_are_close_with` kernels are then generic over the closure `close`,
//! which says whether two values coincide by that variant.
//! So a kernel has no per-value match on the variant.
//!
//! # Const generic flags
//!
//! ## `UNIQUE_Z`
//!
//! Every kernel is generic over `UNIQUE_Z`, or takes it as `unique_z`.
//! If `UNIQUE_Z=true`, the z values are compared too.
//! A runtime `unique_z` is ignored for coordinates without a z.
//!
//! ## `UNIQUE_M`
//!
//! Every kernel is generic over `UNIQUE_M`, or takes it as `unique_m`.
//! If `UNIQUE_M=true`, the m values are compared too.
//! A runtime `unique_m` is ignored for coordinates without an m.
//!
//! ## `NANS_EQUAL`
//!
//! Every kernel is generic over `NANS_EQUAL`, or takes it as `nans_equal`.
//! If `NANS_EQUAL=true`, two NaNs are close.

// TODO: With current implementation, the order matters for MultiPoints, Multi*, and Linestring rings.
// Maybe we should allow giving a bool to first sort?
// For linestrings, it should then first check if it is a ring, and if so ignore one of the repeated
// values.

use std::ops::Range;

use geopolars_core::{
    is_close, is_close_abs, LineStringArray, List, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, Untrimmed, ValidCoords, ValidList,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::ArrowDataType;

use super::has_repeats_in_succession::Coincidence;
use crate::per_row::per_row_bool;

#[inline(always)]
fn values_are_close<const NANS_EQUAL: bool>(
    a: &[f64],
    b: &[f64],
    close: impl Fn(f64, f64) -> bool,
) -> bool {
    a.iter()
        .zip(b)
        .all(|(&a, &b)| close(a, b) || (NANS_EQUAL && a.is_nan() && b.is_nan()))
}

/// Whether there are as many coordinates in `a_vertices` as in `b_vertices`,
/// and every one of them is close to the one at the same position in the other.
#[inline(always)]
fn coordinates_are_close<const UNIQUE_Z: bool, const UNIQUE_M: bool, const NANS_EQUAL: bool>(
    a: &ValidCoords,
    a_vertices: Range<usize>,
    b: &ValidCoords,
    b_vertices: Range<usize>,
    close: impl Fn(f64, f64) -> bool + Copy,
) -> bool {
    let axis_is_close = |a: &[f64], b: &[f64]| {
        values_are_close::<NANS_EQUAL>(&a[a_vertices.clone()], &b[b_vertices.clone()], close)
    };
    a_vertices.len() == b_vertices.len()
        && axis_is_close(a.x(), b.x())
        && axis_is_close(a.y(), b.y())
        && (!UNIQUE_Z
            || axis_is_close(
                a.z().expect("a z, with `UNIQUE_Z`"),
                b.z().expect("a z, with `UNIQUE_Z`"),
            ))
        && (!UNIQUE_M
            || axis_is_close(
                a.m().expect("an m, with `UNIQUE_M`"),
                b.m().expect("an m, with `UNIQUE_M`"),
            ))
}

#[inline(always)]
fn lists_match(a: &ValidList, a_lists: Range<usize>, b: &ValidList, b_lists: Range<usize>) -> bool {
    a_lists.len() == b_lists.len()
        && a_lists
            .zip(b_lists)
            .all(|(i, j)| a.range(i).len() == b.range(j).len())
}

/// Where both rows are there: `a`'s, and the row of `b` it is compared with.
fn compared_validity(a: Option<&Bitmap>, b: Option<&Bitmap>, len: usize) -> Option<Bitmap> {
    match b {
        // A single row of `b`, against every row of `a`.
        Some(b) if b.len() == 1 && len != 1 => match b.get_bit(0) {
            true => a.cloned(),
            false => Some(Bitmap::new_zeroed(len)),
        },
        _ => combine_validities_and(a, b),
    }
}

/// `test(i, j)` of every row `i` of `a`, with `j` the row of `b` it is compared with.
fn per_compared_row(a: &List, b: &List, test: impl Fn(usize, usize) -> bool) -> BooleanArray {
    assert!(
        b.len() == a.len() || b.len() == 1,
        "`b` has to be as long as `a`, or a single row"
    );
    let rows = a
        .clone()
        .with_validity(compared_validity(a.validity(), b.validity(), a.len()));
    if b.len() == a.len() {
        per_row_bool(&rows, |i| test(i, i))
    } else {
        per_row_bool(&rows, |i| test(i, 0))
    }
}

fn points_are_close_with<const UNIQUE_Z: bool, const UNIQUE_M: bool, const NANS_EQUAL: bool>(
    a: &PointArray,
    b: &PointArray,
    close: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    let (a, b) = (&a.coords, &b.coords);
    let len = a.len();
    assert!(
        b.len() == len || b.len() == 1,
        "`b` has to be as long as `a`, or a single row"
    );
    let (a_coords, b_coords) = (a.ignoring_validity(), b.ignoring_validity());
    let rows_are_close = |i: usize, j: usize| {
        coordinates_are_close::<UNIQUE_Z, UNIQUE_M, NANS_EQUAL>(
            a_coords,
            i..i + 1,
            b_coords,
            j..j + 1,
            close,
        )
    };
    // A null point is compared too, but its result is masked by the validity.
    let values = if b.len() == len {
        Bitmap::from_iter((0..len).map(|i| rows_are_close(i, i)))
    } else {
        Bitmap::from_iter((0..len).map(|i| rows_are_close(i, 0)))
    };
    let validity = compared_validity(a.validity(), b.validity(), len);
    BooleanArray::new(ArrowDataType::Boolean, values, validity)
}

fn line_strings_are_close_with<
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
    const NANS_EQUAL: bool,
>(
    a: &Untrimmed<LineStringArray>,
    b: &Untrimmed<LineStringArray>,
    close: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    per_compared_row(a.rows(), b.rows(), |i, j| {
        coordinates_are_close::<UNIQUE_Z, UNIQUE_M, NANS_EQUAL>(
            a.coords(),
            a.rows().range(i),
            b.coords(),
            b.rows().range(j),
            close,
        )
    })
}

fn multi_points_are_close_with<
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
    const NANS_EQUAL: bool,
>(
    a: &Untrimmed<MultiPointArray>,
    b: &Untrimmed<MultiPointArray>,
    close: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    per_compared_row(a.rows(), b.rows(), |i, j| {
        coordinates_are_close::<UNIQUE_Z, UNIQUE_M, NANS_EQUAL>(
            a.coords(),
            a.rows().range(i),
            b.coords(),
            b.rows().range(j),
            close,
        )
    })
}

fn multi_line_strings_are_close_with<
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
    const NANS_EQUAL: bool,
>(
    a: &Untrimmed<MultiLineStringArray>,
    b: &Untrimmed<MultiLineStringArray>,
    close: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    per_compared_row(a.rows(), b.rows(), |i, j| {
        let (a_parts, b_parts) = (a.rows().range(i), b.rows().range(j));
        lists_match(a.lines(), a_parts.clone(), b.lines(), b_parts.clone())
            && coordinates_are_close::<UNIQUE_Z, UNIQUE_M, NANS_EQUAL>(
                a.coords(),
                a.lines().span(a_parts),
                b.coords(),
                b.lines().span(b_parts),
                close,
            )
    })
}

fn polygons_are_close_with<const UNIQUE_Z: bool, const UNIQUE_M: bool, const NANS_EQUAL: bool>(
    a: &Untrimmed<PolygonArray>,
    b: &Untrimmed<PolygonArray>,
    close: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    per_compared_row(a.rows(), b.rows(), |i, j| {
        let (a_rings, b_rings) = (a.rows().range(i), b.rows().range(j));
        lists_match(a.rings(), a_rings.clone(), b.rings(), b_rings.clone())
            && coordinates_are_close::<UNIQUE_Z, UNIQUE_M, NANS_EQUAL>(
                a.coords(),
                a.rings().span(a_rings),
                b.coords(),
                b.rings().span(b_rings),
                close,
            )
    })
}

fn multi_polygons_are_close_with<
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
    const NANS_EQUAL: bool,
>(
    a: &Untrimmed<MultiPolygonArray>,
    b: &Untrimmed<MultiPolygonArray>,
    close: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    per_compared_row(a.rows(), b.rows(), |i, j| {
        let (a_polygons, b_polygons) = (a.rows().range(i), b.rows().range(j));
        if !lists_match(
            a.polygons(),
            a_polygons.clone(),
            b.polygons(),
            b_polygons.clone(),
        ) {
            return false;
        }
        let (a_rings, b_rings) = (a.polygons().span(a_polygons), b.polygons().span(b_polygons));
        lists_match(a.rings(), a_rings.clone(), b.rings(), b_rings.clone())
            && coordinates_are_close::<UNIQUE_Z, UNIQUE_M, NANS_EQUAL>(
                a.coords(),
                a.rings().span(a_rings),
                b.coords(),
                b.rings().span(b_rings),
                close,
            )
    })
}

/// Dispatches a `*_are_close_with` kernel on `unique_z` and `unique_m`
/// (each only for an axis in `$dim`), `nans_equal`, and the variant of `$coincidence`.
macro_rules! are_close {
    ($kernel:ident, $dim:expr, $a:expr, $b:expr, $coincidence:expr, $unique_z:expr, $unique_m:expr, $nans_equal:expr) => {{
        let dim = $dim;
        with_const_flags!(
            UNIQUE_Z = $unique_z && dim.has_z(),
            UNIQUE_M = $unique_m && dim.has_m(),
            NANS_EQUAL = $nans_equal;
            match $coincidence {
                Coincidence::Exact => {
                    $kernel::<UNIQUE_Z, UNIQUE_M, NANS_EQUAL>($a, $b, |a: f64, b: f64| a == b)
                }
                Coincidence::WithinAbs(abs_tol) => {
                    $kernel::<UNIQUE_Z, UNIQUE_M, NANS_EQUAL>($a, $b, move |a, b| {
                        is_close_abs(a, b, abs_tol)
                    })
                }
                Coincidence::WithinRel(tolerance) => {
                    $kernel::<UNIQUE_Z, UNIQUE_M, NANS_EQUAL>($a, $b, move |a, b| {
                        is_close(a, b, tolerance)
                    })
                }
            }
        )
    }};
}

pub fn points_are_close(
    a: &PointArray,
    b: &PointArray,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_close!(
        points_are_close_with,
        a.coords.dim(),
        a,
        b,
        coincidence,
        unique_z,
        unique_m,
        nans_equal
    )
}

pub fn multi_points_are_close(
    a: &Untrimmed<MultiPointArray>,
    b: &Untrimmed<MultiPointArray>,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_close!(
        multi_points_are_close_with,
        a.coords().dim(),
        a,
        b,
        coincidence,
        unique_z,
        unique_m,
        nans_equal
    )
}

pub fn line_strings_are_close(
    a: &Untrimmed<LineStringArray>,
    b: &Untrimmed<LineStringArray>,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_close!(
        line_strings_are_close_with,
        a.coords().dim(),
        a,
        b,
        coincidence,
        unique_z,
        unique_m,
        nans_equal
    )
}

/// The parts also have to be as many, and as long.
pub fn multi_line_strings_are_close(
    a: &Untrimmed<MultiLineStringArray>,
    b: &Untrimmed<MultiLineStringArray>,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_close!(
        multi_line_strings_are_close_with,
        a.coords().dim(),
        a,
        b,
        coincidence,
        unique_z,
        unique_m,
        nans_equal
    )
}

/// The rings also have to be as many, and as long.
pub fn polygons_are_close(
    a: &Untrimmed<PolygonArray>,
    b: &Untrimmed<PolygonArray>,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_close!(
        polygons_are_close_with,
        a.coords().dim(),
        a,
        b,
        coincidence,
        unique_z,
        unique_m,
        nans_equal
    )
}

/// The polygons, and their rings, also have to be as many, and as long.
pub fn multi_polygons_are_close(
    a: &Untrimmed<MultiPolygonArray>,
    b: &Untrimmed<MultiPolygonArray>,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_close!(
        multi_polygons_are_close_with,
        a.coords().dim(),
        a,
        b,
        coincidence,
        unique_z,
        unique_m,
        nans_equal
    )
}
