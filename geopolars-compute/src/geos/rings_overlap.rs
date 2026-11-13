//! Whether two rings overlap.
//! Implements the Python `rings_overlap`.
//!
//! # Const generic flags
//!
//! ## `TOUCHING`
//!
//! Every function here that relates two rings is generic over `TOUCHING`,
//! the Python `include_touching`.

use geopolars_views::{is_ring, LineStringArray, ScalarLineString};
use geos::{Geometry, PreparedGeometry};
use polars_arrow::array::BooleanArray;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::ArrowDataType;

use super::polygons::polygon_enclosed_by_ring_to_geos;
use super::{axes, xy_is_finite_or_error};
use crate::per_row::per_row_optional_bool;
use crate::{row_or_null, GeoComputeError, GeoComputeResult};

/// Their interiors share no area.
const INTERIORS_DISJOINT: &str = "F********";

#[inline(always)]
fn enclosed_areas_overlap<const TOUCHING: bool>(
    a: &PreparedGeometry,
    b: &Geometry,
) -> GeoComputeResult<bool> {
    Ok(if TOUCHING {
        a.intersects(b)?
    } else {
        !a.relate_pattern(b, INTERIORS_DISJOINT)?
    })
}

/// Whether `line` is finite in x and y.
/// Errors if it is, but is not a ring.
#[inline(always)]
fn ring_is_finite(line: &ScalarLineString) -> GeoComputeResult<bool> {
    let (x, y, _, _) = axes(line.coords);
    let vertices = line.vertices.clone();
    if xy_is_finite_or_error(&x[vertices.clone()], &y[vertices.clone()]).is_err() {
        return Ok(false);
    }
    if !is_ring::<false>(line.coords, vertices.clone()) {
        return Err(GeoComputeError::NotARing {
            vertices: vertices.len(),
        });
    }
    Ok(true)
}

/// `None` if one of its vertices is not finite in x or y.
/// Errors if it is not a ring.
#[inline(always)]
fn polygon_enclosed_by_finite_ring(line: &ScalarLineString) -> GeoComputeResult<Option<Geometry>> {
    if !ring_is_finite(line)? {
        return Ok(None);
    }
    polygon_enclosed_by_ring_to_geos(&axes(line.coords), line.vertices.clone()).map(Some)
}

/// Whether rings `a` and `b` overlap.
/// `None` if one of them is not finite in x or y.
/// Errors if one of them is not a ring.
#[inline(always)]
fn ring_pair_overlaps<const TOUCHING: bool>(
    a: &ScalarLineString,
    b: &ScalarLineString,
) -> GeoComputeResult<Option<bool>> {
    let (Some(a), Some(b)) = (
        polygon_enclosed_by_finite_ring(a)?,
        polygon_enclosed_by_finite_ring(b)?,
    ) else {
        return Ok(None);
    };
    let a = PreparedGeometry::new(&a)?;
    enclosed_areas_overlap::<TOUCHING>(&a, &b).map(Some)
}

/// Whether the rings of `a` and `b` overlap, row by row.
///
/// `a` and `b` must have the same length.
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
/// Overlap goes both ways, so this serves a single ring on either side of [rings_overlap_pairwise].
/// `one` is built and prepared once, rather than on every row.
pub fn rings_overlap_against_one<const TOUCHING: bool>(
    one: &ScalarLineString,
    many: &LineStringArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<BooleanArray> {
    let Some(one) = row_or_null(polygon_enclosed_by_finite_ring(one), ignore_per_row_errors)?
    else {
        return Ok(BooleanArray::new_null(
            ArrowDataType::Boolean,
            many.rows.len(),
        ));
    };
    let one = PreparedGeometry::new(&one)?;
    per_row_optional_bool(&many.rows, |i| {
        let ring = row_or_null(
            polygon_enclosed_by_finite_ring(&many.get(i)),
            ignore_per_row_errors,
        )?;
        ring.map(|ring| enclosed_areas_overlap::<TOUCHING>(&one, &ring))
            .transpose()
    })
}
