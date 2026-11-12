use geopolars_compute::geos::{
    interpolate_line_strings, interpolate_multi_line_strings, BufferDistance, InterpolateMode,
};
use geopolars_views::{downcast, Kind, LineStringArray, MultiLineStringArray, PointArray};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::{refused, IntoPolarsResult};

/// Finds the point at `distance` along every linestring or multilinestring
/// in one chunk of a `kind` column's storage, through GEOS.
/// Implements the Python `line_interpolate_point`.
pub fn interpolated_points(
    chunk: &dyn Array,
    kind: Kind,
    distance: BufferDistance,
    mode: InterpolateMode,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> PolarsResult<PointArray> {
    let (ignore, early) = (ignore_per_row_errors, early_finite_check);
    match kind {
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            interpolate_line_strings(&array, distance, mode, ignore, early)
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            interpolate_multi_line_strings(&array, distance, mode, ignore, early)
        }
        Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon => {
            return Err(refused("line_interpolate_point", kind))
        }
    }
    .into_polars()
}
