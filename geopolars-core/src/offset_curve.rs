use geopolars_compute::geos::{
    offset_curves_into_line_strings, offset_curves_into_multi_line_strings, BufferDistance,
    OffsetCurveStyle,
};
use geopolars_compute::{EarlyCoordCheck, OnRowError};
use geopolars_views::{downcast, GeometryArray, LineStringArray, MultiLineStringArray};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::{IntoPolarsResult, LineKind};

/// Finds the offset curve of every linestring or multilinestring
/// in one chunk of a `kind` column's storage, through GEOS.
/// Implements the Python `offset_curve`.
/// The result is XY linestrings.
/// It is XY multilinestrings for a multilinestring, or if `into_multi=true`.
pub fn offset_curves(
    chunk: &dyn Array,
    kind: LineKind,
    into_multi: bool,
    distance: BufferDistance,
    style: &OffsetCurveStyle,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> PolarsResult<GeometryArray> {
    let early = early_coord_check;
    match (kind, into_multi) {
        (LineKind::LineString, false) => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            offset_curves_into_line_strings(array, distance, style, on_row_error, early)
                .map(GeometryArray::LineString)
        }
        (LineKind::LineString, true) => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            let array = GeometryArray::LineString(array);
            offset_curves_into_multi_line_strings(array, distance, style, on_row_error, early)
                .map(GeometryArray::MultiLineString)
        }
        (LineKind::MultiLineString, _) => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            let array = GeometryArray::MultiLineString(array);
            offset_curves_into_multi_line_strings(array, distance, style, on_row_error, early)
                .map(GeometryArray::MultiLineString)
        }
    }
    .into_polars()
}
