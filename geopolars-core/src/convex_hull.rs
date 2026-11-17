use geopolars_compute::geos::{
    line_string_convex_hulls, multi_line_string_convex_hulls, multi_point_convex_hulls,
    multi_polygon_convex_hulls, polygon_convex_hulls,
};
use geopolars_compute::{EarlyCoordCheck, NewHoles, NewRingOrientation, OnRowError};
use geopolars_views::{
    downcast, GeometryArray, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::{IntoPolarsResult, NestedKind};

/// Finds the convex hull of every geometry in one chunk of a `kind` column's storage.
/// Implements the Python `convex_hull`.
pub fn convex_hulls(
    chunk: &dyn Array,
    kind: NestedKind,
    on_row_error: OnRowError,
    new_ring_orientation: NewRingOrientation,
    new_holes: NewHoles,
    early_coord_check: EarlyCoordCheck,
) -> PolarsResult<GeometryArray> {
    let early = early_coord_check;
    match kind {
        NestedKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            line_string_convex_hulls(&array, on_row_error, new_ring_orientation, new_holes, early)
        }
        NestedKind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            multi_point_convex_hulls(&array, on_row_error, new_ring_orientation, new_holes, early)
        }
        NestedKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            multi_line_string_convex_hulls(
                &array,
                on_row_error,
                new_ring_orientation,
                new_holes,
                early,
            )
        }
        NestedKind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            polygon_convex_hulls(&array, on_row_error, new_ring_orientation, new_holes, early)
        }
        NestedKind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            multi_polygon_convex_hulls(&array, on_row_error, new_ring_orientation, new_holes, early)
        }
    }
    .map(GeometryArray::Polygon)
    .into_polars()
}
