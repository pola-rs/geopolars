//! The shortest line between two geometries, forwarded to GEOS.
//!
//! GEOS measures in x and y only, so every output is a `LineStringXY`.

use geopolars_views::output::SizeHint;
use geopolars_views::{GeometryArray, LineStringArray};
use geos::{Geom, Geometry};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::{combine_validities_and, combine_validities_and_not};

use super::{rows_without_vertices, GeosGeometries, GeosRow, GeosRows, ToGeosRows};
use crate::{row_or_null, EarlyCoordCheck, GeoComputeResult, OnRowError};

#[inline(always)]
fn shortest_line_between_rows(
    a: Option<&GeosRow>,
    b: Option<&GeosRow>,
    on_row_error: OnRowError,
) -> GeoComputeResult<Option<GeosRow>> {
    let (Some(a), Some(b)) = (a, b) else {
        return Ok(None);
    };
    match (a, b) {
        (GeosRow::Empty, _) | (_, GeosRow::Empty) => Ok(Some(GeosRow::Empty)),
        (
            GeosRow::Geometry(a) | GeosRow::Unchanged(a),
            GeosRow::Geometry(b) | GeosRow::Unchanged(b),
        ) => row_or_null(
            shortest_line_between_geometries(a, b).map(Some),
            on_row_error,
        ),
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
    lines: impl ExactSizeIterator<Item = GeoComputeResult<Option<GeosRow>>>,
    on_row_error: OnRowError,
) -> GeoComputeResult<LineStringArray> {
    let mut geometries = GeosRows::with_capacity(lines.len());
    for line in lines {
        geometries.push(line?);
    }
    let size_hint = SizeHint {
        coords: 2 * geometries.len(),
        parts: [0, 0],
    };
    GeosGeometries {
        geometries,
        size_hint,
    }
    .into_line_string_array::<false>(on_row_error)
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
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<LineStringArray> {
    debug_assert_eq!(a.len(), b.len(), "`a` and `b` must have the same length");
    let valid = combine_validities_and(a.validity(), b.validity());
    let empty = &rows_without_vertices(&a) | &rows_without_vertices(&b);
    let sent_to_geos = combine_validities_and_not(valid.as_ref(), Some(&empty));
    let early = early_coord_check;
    let a = a
        .with_validity(sent_to_geos.clone())
        .to_geos_rows(on_row_error, early)?;
    let b = b
        .with_validity(sent_to_geos)
        .to_geos_rows(on_row_error, early)?;
    let lines = a
        .geometries
        .iter()
        .zip(b.geometries.iter())
        .enumerate()
        .map(|(i, (a, b))| {
            // Every row that was not sent to GEOS is null on both sides,
            // so an empty one gets its line here.
            if empty.get_bit(i) && is_valid(valid.as_ref(), i) {
                return Ok(Some(GeosRow::Empty));
            }
            shortest_line_between_rows(a, b, on_row_error)
        });
    shortest_lines_into_array(lines, on_row_error)
}

/// Gets the shortest line between a single geometry and every row of `many`,
/// as [shortest_lines_pairwise] does for a `one` repeated on every row.
///
/// `one` must hold a single row.
pub fn shortest_lines_against_one(
    one: &GeometryArray,
    many: &GeometryArray,
    lines_start_on_one: bool,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<LineStringArray> {
    let one = one.to_geos_rows(on_row_error, early_coord_check)?;
    let mut one = one.geometries.iter();
    let (Some(one), None) = (one.next(), one.next()) else {
        panic!("`one` must be a single row");
    };
    let one_is_empty = match one {
        None => {
            let lines = (0..many.len()).map(|_| Ok(None));
            return shortest_lines_into_array(lines, on_row_error);
        }
        Some(GeosRow::Empty) => true,
        Some(GeosRow::Geometry(one) | GeosRow::Unchanged(one)) => one.is_empty()?,
    };
    if one_is_empty {
        let lines =
            (0..many.len()).map(|i| Ok(is_valid(many.validity(), i).then_some(GeosRow::Empty)));
        return shortest_lines_into_array(lines, on_row_error);
    }
    let many = many.to_geos_rows(on_row_error, early_coord_check)?;
    let lines = many.geometries.iter().map(|row| {
        let (a, b) = if lines_start_on_one {
            (one, row)
        } else {
            (row, one)
        };
        shortest_line_between_rows(a, b, on_row_error)
    });
    shortest_lines_into_array(lines, on_row_error)
}
