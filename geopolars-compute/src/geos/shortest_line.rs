//! The shortest line between two geometries, forwarded to GEOS.
//!
//! GEOS measures in x and y only, so every output is a `LineStringXY`.

use geopolars_views::output::SizeHint;
use geopolars_views::{GeometryArray, LineStringArray};
use geos::{Geom, Geometry};

use super::pairs::{GeosPairs, PairOperation, WhenEmpty};
use super::{GeometryOrientation, GeosGeometries, GeosRow, GeosRows};
use crate::{EarlyCoordCheck, GeoComputeResult, OnRowError};

/// See [`shortest_lines`]. An empty row gives an empty line.
struct ShortestLine;

impl PairOperation for ShortestLine {
    type Output = GeosRow;

    fn when_both_empty(&self) -> GeosRow {
        GeosRow::Empty
    }

    fn when_a_empty(&self) -> WhenEmpty<Self> {
        WhenEmpty::Gives(|| GeosRow::Empty)
    }

    fn when_b_empty(&self) -> WhenEmpty<Self> {
        WhenEmpty::Gives(|| GeosRow::Empty)
    }

    fn on_geometries(&self, a: &Geometry, b: &Geometry) -> GeoComputeResult<GeosRow> {
        shortest_line_between_geometries(a, b)
    }
}

#[inline(always)]
fn shortest_line_between_geometries(a: &Geometry, b: &Geometry) -> GeoComputeResult<GeosRow> {
    // GEOS gives no nearest points for a geometry it calls empty,
    // which `nearest_points` reports as an error.
    if a.is_empty()? || b.is_empty()? {
        return Ok(GeosRow::Empty);
    }
    let line = Geometry::create_line_string(a.nearest_points(b)?)?;
    Ok(GeosRow::Geometry {
        geometry: line,
        maybe_wrong_orientation: false,
    })
}

fn shortest_lines_into_array(
    len: usize,
    lines: impl Iterator<Item = GeoComputeResult<Option<GeosRow>>>,
    on_row_error: OnRowError,
) -> GeoComputeResult<LineStringArray> {
    let mut geometries = GeosRows::with_capacity(len);
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
        // Lines have no rings.
        orientation: GeometryOrientation::UNKNOWN,
    }
    .into_line_string_array::<false>(on_row_error)
}

/// Gets the shortest line from `a` to `b`, row by row.
/// Implements the Python `shortest_line_between`.
/// A single row on either side is paired with every row of the other.
///
/// `a` and `b` must have the same length, or one of them a single row.
pub fn shortest_lines(
    a: &GeometryArray,
    b: &GeometryArray,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<LineStringArray> {
    let pairs = GeosPairs::new(a, b, &ShortestLine, on_row_error, early_coord_check)?;
    let lines = pairs.evaluate(&ShortestLine, on_row_error);
    shortest_lines_into_array(pairs.len(), lines, on_row_error)
}
