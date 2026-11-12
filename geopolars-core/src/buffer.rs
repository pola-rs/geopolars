use geopolars_compute::geos::{
    buffer_into_multi_polygons, buffer_into_polygons, BufferDistance, BufferStyle,
};
use geopolars_views::{GeometryArray, Kind, MultiPolygonArray, PolygonArray};
use polars_arrow::array::{Array, ListArray};
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// Buffers every geometry in one chunk of a `kind` column's storage, through GEOS.
/// Implements the Python `buffer`.
/// The result is the storage of XY multipolygons if `into_multi=true`, and of XY polygons otherwise.
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
) -> PolarsResult<ListArray<i64>> {
    let (ignore, early) = (ignore_per_row_errors, early_finite_check);
    let array = &GeometryArray::new(chunk, kind)?;
    if into_multi {
        buffer_into_multi_polygons(array, distance, style, ccw_outer, ignore, early)
            .map(MultiPolygonArray::into_list_array)
    } else {
        buffer_into_polygons(array, distance, style, ccw_outer, ignore, early)
            .map(PolygonArray::into_list_array)
    }
    .into_polars()
}
