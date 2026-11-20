//! Whether two geometries are equal coordinate by coordinate, exactly or within a tolerance.
//! Implements the Python `is_equal`.
//!
//! Every kernel takes `a`, and `b` of the same length (row by row) or of a single row (against every row of `a`).
//! Every comparison stops at the first part length or value that does not match.
//!
//! # Const generic flags
//!
//! Every kernel is generic over `Z`, `M`, and `NANS_EQUAL`,
//! or takes the Python `include_z`, `include_m`, and `nans_equal` at runtime.
//! A runtime `include_z` or `include_m` leaves its axis out of the compared coordinates
//! (see [ValidCoords::with_only_axes]), and is ignored for coordinates without that axis.
//! `Z` and `M` may only be `true` for an axis the compared coordinates have.

// TODO: With current implementation, the order matters for MultiPoints, Multi*, and Linestring rings.
// Maybe we should allow giving a bool to first sort?
// For linestrings, it should then first check if it is a ring, and if so ignore one of the repeated
// values.

use std::ops::Range;

use geopolars_views::{
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
fn values_coincide<const NANS_EQUAL: bool>(
    a: &[f64],
    b: &[f64],
    coincide: impl Fn(f64, f64) -> bool,
) -> bool {
    a.iter()
        .zip(b)
        .all(|(&a, &b)| coincide(a, b) || (NANS_EQUAL && a.is_nan() && b.is_nan()))
}

/// Tests whether two runs of vertices have the same length, and coincide coordinate by coordinate.
#[inline(always)]
fn coordinates_coincide<const Z: bool, const M: bool, const NANS_EQUAL: bool>(
    a: &ValidCoords,
    a_vertices: Range<usize>,
    b: &ValidCoords,
    b_vertices: Range<usize>,
    coincide: impl Fn(f64, f64) -> bool + Copy,
) -> bool {
    let axis_coincides = |a: &[f64], b: &[f64]| {
        values_coincide::<NANS_EQUAL>(&a[a_vertices.clone()], &b[b_vertices.clone()], coincide)
    };
    a_vertices.len() == b_vertices.len()
        && axis_coincides(a.x(), b.x())
        && axis_coincides(a.y(), b.y())
        && (!Z || axis_coincides(a.z().expect("a z, with `Z`"), b.z().expect("a z, with `Z`")))
        && (!M
            || axis_coincides(
                a.m().expect("an m, with `M`"),
                b.m().expect("an m, with `M`"),
            ))
}

#[inline(always)]
fn lists_match(a: &ValidList, a_lists: Range<usize>, b: &ValidList, b_lists: Range<usize>) -> bool {
    a_lists.len() == b_lists.len()
        && a_lists
            .zip(b_lists)
            .all(|(i, j)| a.range(i).len() == b.range(j).len())
}

/// Combines the validity of every row of `a` with that of the row of `b` it is compared with.
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

/// Runs `test(i, j)` on every row `i` of `a`, with `j` the row of `b` it is compared with.
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

fn points_are_equal_with<const Z: bool, const M: bool, const NANS_EQUAL: bool>(
    a: &PointArray,
    b: &PointArray,
    a_coords: &ValidCoords,
    b_coords: &ValidCoords,
    coincide: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    let (a, b) = (&a.coords, &b.coords);
    let len = a.len();
    assert!(
        b.len() == len || b.len() == 1,
        "`b` has to be as long as `a`, or a single row"
    );
    let rows_are_equal = |i: usize, j: usize| {
        coordinates_coincide::<Z, M, NANS_EQUAL>(a_coords, i..i + 1, b_coords, j..j + 1, coincide)
    };
    // A null point is compared too, but its result is masked by the validity.
    let values = if b.len() == len {
        Bitmap::from_iter((0..len).map(|i| rows_are_equal(i, i)))
    } else {
        Bitmap::from_iter((0..len).map(|i| rows_are_equal(i, 0)))
    };
    let validity = compared_validity(a.validity(), b.validity(), len);
    BooleanArray::new(ArrowDataType::Boolean, values, validity)
}

fn line_strings_are_equal_with<const Z: bool, const M: bool, const NANS_EQUAL: bool>(
    a: &Untrimmed<LineStringArray>,
    b: &Untrimmed<LineStringArray>,
    a_coords: &ValidCoords,
    b_coords: &ValidCoords,
    coincide: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    per_compared_row(a.rows(), b.rows(), |i, j| {
        coordinates_coincide::<Z, M, NANS_EQUAL>(
            a_coords,
            a.rows().range(i),
            b_coords,
            b.rows().range(j),
            coincide,
        )
    })
}

fn multi_points_are_equal_with<const Z: bool, const M: bool, const NANS_EQUAL: bool>(
    a: &Untrimmed<MultiPointArray>,
    b: &Untrimmed<MultiPointArray>,
    a_coords: &ValidCoords,
    b_coords: &ValidCoords,
    coincide: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    per_compared_row(a.rows(), b.rows(), |i, j| {
        coordinates_coincide::<Z, M, NANS_EQUAL>(
            a_coords,
            a.rows().range(i),
            b_coords,
            b.rows().range(j),
            coincide,
        )
    })
}

fn multi_line_strings_are_equal_with<const Z: bool, const M: bool, const NANS_EQUAL: bool>(
    a: &Untrimmed<MultiLineStringArray>,
    b: &Untrimmed<MultiLineStringArray>,
    a_coords: &ValidCoords,
    b_coords: &ValidCoords,
    coincide: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    per_compared_row(a.rows(), b.rows(), |i, j| {
        let (a_parts, b_parts) = (a.rows().range(i), b.rows().range(j));
        lists_match(a.lines(), a_parts.clone(), b.lines(), b_parts.clone())
            && coordinates_coincide::<Z, M, NANS_EQUAL>(
                a_coords,
                a.lines().span(a_parts),
                b_coords,
                b.lines().span(b_parts),
                coincide,
            )
    })
}

fn polygons_are_equal_with<const Z: bool, const M: bool, const NANS_EQUAL: bool>(
    a: &Untrimmed<PolygonArray>,
    b: &Untrimmed<PolygonArray>,
    a_coords: &ValidCoords,
    b_coords: &ValidCoords,
    coincide: impl Fn(f64, f64) -> bool + Copy,
) -> BooleanArray {
    per_compared_row(a.rows(), b.rows(), |i, j| {
        let (a_rings, b_rings) = (a.rows().range(i), b.rows().range(j));
        lists_match(a.rings(), a_rings.clone(), b.rings(), b_rings.clone())
            && coordinates_coincide::<Z, M, NANS_EQUAL>(
                a_coords,
                a.rings().span(a_rings),
                b_coords,
                b.rings().span(b_rings),
                coincide,
            )
    })
}

fn multi_polygons_are_equal_with<const Z: bool, const M: bool, const NANS_EQUAL: bool>(
    a: &Untrimmed<MultiPolygonArray>,
    b: &Untrimmed<MultiPolygonArray>,
    a_coords: &ValidCoords,
    b_coords: &ValidCoords,
    coincide: impl Fn(f64, f64) -> bool + Copy,
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
            && coordinates_coincide::<Z, M, NANS_EQUAL>(
                a_coords,
                a.rings().span(a_rings),
                b_coords,
                b.rings().span(b_rings),
                coincide,
            )
    })
}

/// Dispatches a `*_are_equal_with` kernel on `nans_equal`, the variant of `$coincidence`,
/// and the axes of `$a_coords` and `$b_coords` that `$include_z` and `$include_m` keep.
/// Each variant gets its own closure `coincide`,
/// so the kernel does not match on the variant for every value.
macro_rules! are_equal {
    ($kernel:ident, $a:expr, $a_coords:expr, $b:expr, $b_coords:expr, $coincidence:expr, $include_z:expr, $include_m:expr, $nans_equal:expr) => {{
        let a_coords = &$a_coords.with_only_axes($include_z, $include_m);
        let b_coords = &$b_coords.with_only_axes($include_z, $include_m);
        with_const_flags!(
            Z = a_coords.z().is_some(),
            M = a_coords.m().is_some(),
            NANS_EQUAL = $nans_equal;
            match $coincidence {
                Coincidence::Exact => {
                    $kernel::<Z, M, NANS_EQUAL>($a, $b, a_coords, b_coords, |a: f64, b: f64| a == b)
                }
                Coincidence::WithinAbs(abs_tol) => {
                    $kernel::<Z, M, NANS_EQUAL>($a, $b, a_coords, b_coords, move |a, b| {
                        is_close_abs(a, b, abs_tol)
                    })
                }
                Coincidence::WithinRel(tolerance) => {
                    $kernel::<Z, M, NANS_EQUAL>($a, $b, a_coords, b_coords, move |a, b| {
                        is_close(a, b, tolerance)
                    })
                }
            }
        )
    }};
}

pub fn points_are_equal(
    a: &PointArray,
    b: &PointArray,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_equal!(
        points_are_equal_with,
        a,
        a.coords.ignoring_validity(),
        b,
        b.coords.ignoring_validity(),
        coincidence,
        include_z,
        include_m,
        nans_equal
    )
}

pub fn multi_points_are_equal(
    a: &Untrimmed<MultiPointArray>,
    b: &Untrimmed<MultiPointArray>,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_equal!(
        multi_points_are_equal_with,
        a,
        a.coords(),
        b,
        b.coords(),
        coincidence,
        include_z,
        include_m,
        nans_equal
    )
}

pub fn line_strings_are_equal(
    a: &Untrimmed<LineStringArray>,
    b: &Untrimmed<LineStringArray>,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_equal!(
        line_strings_are_equal_with,
        a,
        a.coords(),
        b,
        b.coords(),
        coincidence,
        include_z,
        include_m,
        nans_equal
    )
}

pub fn multi_line_strings_are_equal(
    a: &Untrimmed<MultiLineStringArray>,
    b: &Untrimmed<MultiLineStringArray>,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_equal!(
        multi_line_strings_are_equal_with,
        a,
        a.coords(),
        b,
        b.coords(),
        coincidence,
        include_z,
        include_m,
        nans_equal
    )
}

pub fn polygons_are_equal(
    a: &Untrimmed<PolygonArray>,
    b: &Untrimmed<PolygonArray>,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_equal!(
        polygons_are_equal_with,
        a,
        a.coords(),
        b,
        b.coords(),
        coincidence,
        include_z,
        include_m,
        nans_equal
    )
}

pub fn multi_polygons_are_equal(
    a: &Untrimmed<MultiPolygonArray>,
    b: &Untrimmed<MultiPolygonArray>,
    coincidence: Coincidence,
    include_z: bool,
    include_m: bool,
    nans_equal: bool,
) -> BooleanArray {
    are_equal!(
        multi_polygons_are_equal_with,
        a,
        a.coords(),
        b,
        b.coords(),
        coincidence,
        include_z,
        include_m,
        nans_equal
    )
}
