use geopolars_compute::geos::{
    offset_curves_into_line_strings, offset_curves_into_multi_line_strings, BufferDistance,
    OffsetCurveStyle,
};
use geopolars_views::{downcast, Kind, LineStringArray, MultiLineStringArray};
use polars_arrow::array::{Array, ListArray};
use polars_core::prelude::PolarsResult;

use crate::{refused, IntoPolarsResult};

/// Finds the offset curve of every linestring or multilinestring
/// in one chunk of a `kind` column's storage, through GEOS.
/// Implements the Python `offset_curve`.
/// The result is the storage of XY multilinestrings for a multilinestring or if `into_multi=true`,
/// and of XY linestrings otherwise.
pub fn offset_curves(
    chunk: &dyn Array,
    kind: Kind,
    into_multi: bool,
    distance: BufferDistance,
    style: &OffsetCurveStyle,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> PolarsResult<ListArray<i64>> {
    let (ignore, early) = (ignore_per_row_errors, early_finite_check);
    match (kind, into_multi) {
        (Kind::LineString, false) => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            offset_curves_into_line_strings(&array, distance, style, ignore, early)
                .map(LineStringArray::into_list_array)
        }
        (Kind::LineString, true) => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            offset_curves_into_multi_line_strings(&array, distance, style, ignore, early)
                .map(MultiLineStringArray::into_list_array)
        }
        (Kind::MultiLineString, _) => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            offset_curves_into_multi_line_strings(&array, distance, style, ignore, early)
                .map(MultiLineStringArray::into_list_array)
        }
        (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon, _) => {
            return Err(refused("offset_curve", kind))
        }
    }
    .into_polars()
}
