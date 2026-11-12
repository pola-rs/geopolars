//! Whether two rings overlap, which is when the areas they enclose share area.
//!
//! # Const generic flags
//!
//! ## `TOUCHING`
//!
//! Every function here that relates two rings is generic over `TOUCHING`.
//! If `TOUCHING=true`, rings that only touch overlap too.

use ::rsgeo::coordinate_position::CoordPos;
use ::rsgeo::dimensions::Dimensions;
use ::rsgeo::relate::IntersectionMatrix;
use ::rsgeo::{Polygon, PreparedGeometry, Relate};
use geopolars_views::{LineStringArray, ScalarLineString};
use polars_arrow::array::BooleanArray;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::ArrowDataType;

use super::{per_row_optional_bool, ring_is_finite, to_rsgeo_ring};
use crate::{row_or_null, GeoComputeResult};

/// Whether two rings related as `matrix` overlap.
#[inline(always)]
pub(super) fn overlap<const TOUCHING: bool>(matrix: &IntersectionMatrix) -> bool {
    if TOUCHING {
        matrix.is_intersects()
    } else {
        matrix.get(CoordPos::Inside, CoordPos::Inside) != Dimensions::Empty
    }
}

/// The polygon the ring `line` encloses, in x and y.
/// `None` if one of its vertices is not finite in x or y.
/// Errors if it is not a ring.
#[inline(always)]
fn finite_ring(line: &ScalarLineString) -> GeoComputeResult<Option<Polygon<f64>>> {
    Ok(ring_is_finite(line)?.then(|| to_rsgeo_ring(line)))
}

/// Whether rings `a` and `b` overlap.
/// `None` if one of them is not finite in x or y.
/// Errors if one of them is not a ring.
#[inline(always)]
fn ring_pair_overlaps<const TOUCHING: bool>(
    a: &ScalarLineString,
    b: &ScalarLineString,
) -> GeoComputeResult<Option<bool>> {
    let (Some(a), Some(b)) = (finite_ring(a)?, finite_ring(b)?) else {
        return Ok(None);
    };
    Ok(Some(overlap::<TOUCHING>(&a.relate(&b))))
}

/// Whether the rings of `a` and `b`, which have the same length, overlap row by row.
/// A row is null where either ring is, or where one of them is not finite in x or y.
/// If `ignore_per_row_errors=true`, a row where one of them is not a ring is null too;
/// otherwise that is the error.
pub fn rings_overlap_pairwise<const TOUCHING: bool>(
    a: &LineStringArray,
    b: &LineStringArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<BooleanArray> {
    debug_assert_eq!(a.rows.len(), b.rows.len(), "Chunks must be the same size");
    let rows = a
        .rows
        .clone()
        .with_validity(combine_validities_and(a.rows.validity(), b.rows.validity()));
    per_row_optional_bool(&rows, |i| {
        row_or_null(
            ring_pair_overlaps::<TOUCHING>(&a.get(i), &b.get(i)),
            ignore_per_row_errors,
        )
    })
}

/// Whether the single ring `one` overlaps each ring of `many`.
/// Overlap goes both ways, so this serves a single ring on either side.
/// `one` is copied for rsgeo and prepared once, rather than on every row.
/// See [rings_overlap_pairwise] for `ignore_per_row_errors`, and what is null.
/// Every row is null where `one` would make it so.
pub fn rings_overlap_against_one<const TOUCHING: bool>(
    one: &ScalarLineString,
    many: &LineStringArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<BooleanArray> {
    let Some(one) = row_or_null(finite_ring(one), ignore_per_row_errors)? else {
        return Ok(BooleanArray::new_null(
            ArrowDataType::Boolean,
            many.rows.len(),
        ));
    };
    let one = PreparedGeometry::from(one);
    per_row_optional_bool(&many.rows, |i| {
        let ring = row_or_null(finite_ring(&many.get(i)), ignore_per_row_errors)?;
        Ok(ring.map(|ring| overlap::<TOUCHING>(&one.relate(&ring))))
    })
}
