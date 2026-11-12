use geopolars_compute::geos::{
    line_string_convex_hulls, multi_line_string_convex_hulls, multi_point_convex_hulls,
    multi_polygon_convex_hulls, polygon_convex_hulls,
};
use geopolars_views::{
    downcast, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PolygonArray,
};
use polars_arrow::array::{Array, ListArray};
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
) -> PolarsResult<PolygonArray> {
    let rows = downcast::<ListArray<i64>>(chunk, "a list of parts")?;
    let (ignore, early) = (ignore_per_row_errors, early_finite_check);
    match kind {
        NestedKind::LineString => {
            line_string_convex_hulls(&LineStringArray::new(rows)?, ignore, ccw_outer, early)
        }
        NestedKind::MultiPoint => {
            multi_point_convex_hulls(&MultiPointArray::new(rows)?, ignore, ccw_outer, early)
        }
        NestedKind::MultiLineString => multi_line_string_convex_hulls(
            &MultiLineStringArray::new(rows)?,
            ignore,
            ccw_outer,
            early,
        ),
        NestedKind::Polygon => {
            polygon_convex_hulls(&PolygonArray::new(rows)?, ignore, ccw_outer, early)
        }
        NestedKind::MultiPolygon => {
            multi_polygon_convex_hulls(&MultiPolygonArray::new(rows)?, ignore, ccw_outer, early)
        }
    }
    .into_polars()
}
