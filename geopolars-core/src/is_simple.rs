use geopolars_compute::geos::{
    line_strings_are_simple, multi_line_strings_are_simple, multi_polygons_are_simple,
    polygons_are_simple,
};
use geopolars_compute::native::{multi_points_are_simple, points_are_simple};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// Checks whether every geometry in one chunk of a `kind` column's storage is simple:
/// natively for (multi)points, through GEOS for the other kinds.
/// Implements the Python `is_simple`.
pub fn are_simple(chunk: &dyn Array, kind: Kind) -> PolarsResult<BooleanArray> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_are_simple(&PointArray::new(Coords::new(chunk)?)),
        Kind::MultiPoint => multi_points_are_simple(&MultiPointArray::new(rows()?)?),
        Kind::LineString => {
            line_strings_are_simple(&LineStringArray::new(rows()?)?).into_polars()?
        }
        Kind::MultiLineString => {
            multi_line_strings_are_simple(&MultiLineStringArray::new(rows()?)?).into_polars()?
        }
        Kind::Polygon => polygons_are_simple(&PolygonArray::new(rows()?)?).into_polars()?,
        Kind::MultiPolygon => {
            multi_polygons_are_simple(&MultiPolygonArray::new(rows()?)?).into_polars()?
        }
    })
}
