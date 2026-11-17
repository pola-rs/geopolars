use geopolars_compute::geos::{
    simplify_line_strings, simplify_multi_line_strings, simplify_multi_polygons, simplify_polygons,
};
use geopolars_compute::{EarlyCoordCheck, OnRowError};
use geopolars_views::{
    downcast, GeometryArray, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::{IntoPolarsResult, SegmentKind};

/// Simplifies every (multi)linestring or (multi)polygon through GEOS,
/// for one chunk of a `kind` column's storage.
/// Implements the Python `simplify`.
/// The result is of the same kind, with the input's z but without m.
pub fn simplify(
    chunk: &dyn Array,
    kind: SegmentKind,
    tolerance: f64,
    preserve_topology: bool,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> PolarsResult<GeometryArray> {
    let early = early_coord_check;
    match kind {
        SegmentKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            simplify_line_strings(&array, tolerance, preserve_topology, on_row_error, early)
                .map(GeometryArray::LineString)
        }
        SegmentKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            simplify_multi_line_strings(&array, tolerance, preserve_topology, on_row_error, early)
                .map(GeometryArray::MultiLineString)
        }
        SegmentKind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            simplify_polygons(&array, tolerance, preserve_topology, on_row_error, early)
                .map(GeometryArray::Polygon)
        }
        SegmentKind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            simplify_multi_polygons(&array, tolerance, preserve_topology, on_row_error, early)
                .map(GeometryArray::MultiPolygon)
        }
    }
    .into_polars()
}
