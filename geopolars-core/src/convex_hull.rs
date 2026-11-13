use geopolars_compute::geos::{
    line_string_convex_hulls, multi_line_string_convex_hulls, multi_point_convex_hulls,
    multi_polygon_convex_hulls, polygon_convex_hulls,
};
use geopolars_views::{
    downcast, GeometryArray, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::{IntoPolarsResult, NestedKind};

/// Finds the convex hull of every geometry in one chunk of a `kind` column's storage,
/// through GEOS.
/// Implements the Python `convex_hull`.
pub fn convex_hulls(
    chunk: &dyn Array,
    kind: NestedKind,
    ignore_per_row_errors: bool,
    ccw_outer: bool,
    early_finite_check: bool,
) -> PolarsResult<GeometryArray> {
    let (ignore, early) = (ignore_per_row_errors, early_finite_check);
    match kind {
        NestedKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            line_string_convex_hulls(&array, ignore, ccw_outer, early)
        }
        NestedKind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            multi_point_convex_hulls(&array, ignore, ccw_outer, early)
        }
        NestedKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            multi_line_string_convex_hulls(&array, ignore, ccw_outer, early)
        }
        NestedKind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            polygon_convex_hulls(&array, ignore, ccw_outer, early)
        }
        NestedKind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            multi_polygon_convex_hulls(&array, ignore, ccw_outer, early)
        }
    }
    .map(GeometryArray::Polygon)
    .into_polars()
}
