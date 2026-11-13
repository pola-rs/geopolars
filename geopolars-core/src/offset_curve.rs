use geopolars_compute::geos::{
    offset_curves_into_line_strings, offset_curves_into_multi_line_strings, BufferDistance,
    OffsetCurveStyle,
};
use geopolars_views::{downcast, GeometryArray, LineStringArray, MultiLineStringArray};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::{IntoPolarsResult, LineKind};

/// Finds the offset curve of every linestring or multilinestring
/// in one chunk of a `kind` column's storage, through GEOS.
/// Implements the Python `offset_curve`.
/// The result is XY multilinestrings for a multilinestring or if `into_multi=true`,
/// and XY linestrings otherwise.
pub fn offset_curves(
    chunk: &dyn Array,
    kind: LineKind,
    into_multi: bool,
    distance: BufferDistance,
    style: &OffsetCurveStyle,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> PolarsResult<GeometryArray> {
    let (ignore, early) = (ignore_per_row_errors, early_finite_check);
    match (kind, into_multi) {
        (LineKind::LineString, false) => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            offset_curves_into_line_strings(&array, distance, style, ignore, early)
                .map(GeometryArray::LineString)
        }
        (LineKind::LineString, true) => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            offset_curves_into_multi_line_strings(&array, distance, style, ignore, early)
                .map(GeometryArray::MultiLineString)
        }
        (LineKind::MultiLineString, _) => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            offset_curves_into_multi_line_strings(&array, distance, style, ignore, early)
                .map(GeometryArray::MultiLineString)
        }
    }
    .into_polars()
}
