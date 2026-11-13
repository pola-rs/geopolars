//! The shortest line between two geometries, forwarded to GEOS.
//!
//! GEOS measures in x and y only, so every output is a `LineStringXY`.

use geopolars_views::output::SizeHint;
use geopolars_views::{GeometryArray, LineStringArray};
use geos::{Geom, Geometry};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::{combine_validities_and, combine_validities_and_not};

use super::{rows_without_vertices, GeosGeometries, GeosRow, ToGeosRows};
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
    // A geometry with parts can still be empty, such as a polygon whose only ring has no vertices.
    // GEOS gives no nearest points for it, which `nearest_points` reports as an error.
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

fn is_valid(validity: Option<&Bitmap>, i: usize) -> bool {
    validity.is_none_or(|validity| validity.get_bit(i))
}

/// Gets the shortest line from `a` to `b`, row by row.
/// Implements the Python `shortest_line_between`.
///
/// `a` and `b` must have the same length.
pub fn shortest_lines_pairwise(
    a: GeometryArray,
    b: GeometryArray,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<LineStringArray> {
    debug_assert_eq!(a.len(), b.len(), "`a` and `b` must have the same length");
    let valid = combine_validities_and(a.validity(), b.validity());
    let empty = &rows_without_vertices(&a) | &rows_without_vertices(&b);
    let sent_to_geos = combine_validities_and_not(valid.as_ref(), Some(&empty));
    let (ignore, early) = (ignore_per_row_errors, early_finite_check);
    let a = a
        .with_validity(sent_to_geos.clone())
        .to_geos_rows(ignore, early)?;
    let b = b.with_validity(sent_to_geos).to_geos_rows(ignore, early)?;
    let lines = a
        .geometries
        .iter()
        .zip(&b.geometries)
        .enumerate()
        .map(|(i, (a, b))| {
            // Every row that was not sent to GEOS is null on both sides,
            // so an empty one gets its line here.
            if empty.get_bit(i) && is_valid(valid.as_ref(), i) {
                return Ok(GeosRow::Empty);
            }
            shortest_line_between_rows(a, b, ignore_per_row_errors)
        })
        .collect::<GeoComputeResult<Vec<_>>>()?;
    shortest_lines_into_array(lines, ignore_per_row_errors)
}

/// Gets the shortest line between a single geometry and every row of `many`,
/// as [shortest_lines_pairwise] does for a `one` repeated on every row.
///
/// `one` must hold a single row.
pub fn shortest_lines_against_one(
    one: &GeometryArray,
    many: &GeometryArray,
    lines_start_on_one: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<LineStringArray> {
    let one = one.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let [one] = one.geometries.as_slice() else {
        panic!("`one` must be a single row, got {}", one.geometries.len());
    };
    let one_is_empty = match one {
        GeosRow::Null => {
            let lines = (0..many.len()).map(|_| GeosRow::Null).collect();
            return shortest_lines_into_array(lines, ignore_per_row_errors);
        }
        GeosRow::Empty => true,
        GeosRow::Geometry(one) => one.is_empty()?,
    };
    if one_is_empty {
        let lines = (0..many.len())
            .map(|i| match is_valid(many.validity(), i) {
                true => GeosRow::Empty,
                false => GeosRow::Null,
            })
            .collect();
        return shortest_lines_into_array(lines, ignore_per_row_errors);
    }
    let many = many.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let lines = many
        .geometries
        .iter()
        .map(|row| {
            let (a, b) = if lines_start_on_one {
                (one, row)
            } else {
                (row, one)
            };
            shortest_line_between_rows(a, b, ignore_per_row_errors)
        })
        .collect::<GeoComputeResult<Vec<_>>>()?;
    shortest_lines_into_array(lines, ignore_per_row_errors)
}
