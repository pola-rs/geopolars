//! The shortest line between two geometries, forwarded to GEOS.
//!
//! GEOS measures in x and y only, so every output is a `LineStringXY`.

use geopolars_core::output::SizeHint;
use geopolars_core::{GeometryArray, LineStringArray};
use geos::{Geom, Geometry};

use super::{GeosGeometries, GeosRow, ToGeosRows};
use crate::GeoComputeResult;

#[inline(always)]
fn shortest_line_between_rows(
    a: &GeosRow,
    b: &GeosRow,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<GeosRow> {
    match (a, b) {
        (GeosRow::Null, _) | (_, GeosRow::Null) => Ok(GeosRow::Null),
        (GeosRow::Empty, _) | (_, GeosRow::Empty) => Ok(GeosRow::Empty),
        (GeosRow::Geometry(a), GeosRow::Geometry(b)) => {
            match shortest_line_between_geometries(a, b) {
                Err(_) if ignore_per_row_errors => Ok(GeosRow::Null),
                line => line,
            }
        }
    }
}

#[inline(always)]
fn shortest_line_between_geometries(a: &Geometry, b: &Geometry) -> GeoComputeResult<GeosRow> {
    // GEOS fails on an empty geometry, rather than giving no nearest points.
    if a.is_empty()? || b.is_empty()? {
        return Ok(GeosRow::Empty);
    }
    let line = Geometry::create_line_string(a.nearest_points(b)?)?;
    Ok(GeosRow::Geometry(line))
}

fn shortest_lines_into_array(
    lines: Vec<GeosRow>,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<LineStringArray> {
    let size_hint = SizeHint {
        coords: 2 * lines.len(),
        parts: [0, 0],
    };
    GeosGeometries {
        geometries: lines,
        size_hint,
    }
    .into_line_string_array::<false>(ignore_per_row_errors)
}

/// Gets the shortest line from `a` to `b`, row by row.
/// Implements the Python `shortest_line_between`, which documents its behaviour.
/// Each side is converted to GEOS in full, so a row that is null on only one side can still fail.
///
/// `a` and `b` must have the same length.
pub fn shortest_lines_pairwise(
    a: &GeometryArray,
    b: &GeometryArray,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<LineStringArray> {
    let a = a.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let b = b.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    debug_assert_eq!(
        a.geometries.len(),
        b.geometries.len(),
        "`a` and `b` must have the same length"
    );
    let lines = a
        .geometries
        .iter()
        .zip(&b.geometries)
        .map(|(a, b)| shortest_line_between_rows(a, b, ignore_per_row_errors))
        .collect::<GeoComputeResult<Vec<_>>>()?;
    shortest_lines_into_array(lines, ignore_per_row_errors)
}

/// Gets the shortest line between a single geometry and every row of `many`.
/// The lines start on `one` if `one_is_a=true`, and end on it if `one_is_a=false`.
/// As in [shortest_lines_pairwise], `many` is converted to GEOS in full, even if `one` is null.
///
/// `one` must hold a single row.
pub fn shortest_lines_against_one(
    one: &GeometryArray,
    many: &GeometryArray,
    one_is_a: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<LineStringArray> {
    let one = one.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let [one] = one.geometries.as_slice() else {
        panic!("`one` must be a single row, got {}", one.geometries.len());
    };
    let many = many.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let lines = many
        .geometries
        .iter()
        .map(|row| {
            let (a, b) = if one_is_a { (one, row) } else { (row, one) };
            shortest_line_between_rows(a, b, ignore_per_row_errors)
        })
        .collect::<GeoComputeResult<Vec<_>>>()?;
    shortest_lines_into_array(lines, ignore_per_row_errors)
}
