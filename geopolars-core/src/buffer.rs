use geopolars_compute::geos::{
    buffer_into_multi_polygons, buffer_into_polygons, BufferDistance, BufferStyle,
};
use geopolars_views::{GeometryArray, Kind};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// Buffers every geometry in one chunk of a `kind` column's storage, through GEOS.
/// Implements the Python `buffer`.
/// The result is XY polygons, or XY multipolygons if `into_multi=true`.
#[allow(clippy::too_many_arguments)]
pub fn buffers(
    chunk: &dyn Array,
    kind: Kind,
    into_multi: bool,
    distance: BufferDistance,
    style: &BufferStyle,
    ccw_outer: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> PolarsResult<GeometryArray> {
    let (ignore, early) = (ignore_per_row_errors, early_finite_check);
    let array = &GeometryArray::new(chunk, kind)?;
    if into_multi {
        buffer_into_multi_polygons(array, distance, style, ccw_outer, ignore, early)
            .map(GeometryArray::MultiPolygon)
    } else {
        buffer_into_polygons(array, distance, style, ccw_outer, ignore, early)
            .map(GeometryArray::Polygon)
    }
    .into_polars()
}
