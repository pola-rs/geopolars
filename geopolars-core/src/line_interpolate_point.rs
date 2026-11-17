use geopolars_compute::geos::{
    interpolate_line_strings, interpolate_multi_line_strings, BufferDistance, InterpolateMode,
};
use geopolars_compute::OnRowError;
use geopolars_views::{downcast, GeometryArray, LineStringArray, MultiLineStringArray};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::{IntoPolarsResult, LineKind};

/// Finds the point at `distance` along every linestring or multilinestring
/// in one chunk of a `kind` column's storage, through GEOS.
/// Implements the Python `line_interpolate_point`.
pub fn interpolated_points(
    chunk: &dyn Array,
    kind: LineKind,
    distance: BufferDistance,
    mode: InterpolateMode,
    on_row_error: OnRowError,
    early_finite_check: bool,
) -> PolarsResult<GeometryArray> {
    let early = early_finite_check;
    match kind {
        LineKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            interpolate_line_strings(&array, distance, mode, on_row_error, early)
        }
        LineKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            interpolate_multi_line_strings(&array, distance, mode, on_row_error, early)
        }
    }
    .map(GeometryArray::Point)
    .into_polars()
}
