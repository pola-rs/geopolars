use geopolars_compute::native::{
    line_strings_are_equal, multi_line_strings_are_equal, multi_points_are_equal,
    multi_polygons_are_equal, points_are_equal, polygons_are_equal, Coincidence,
};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_core::prelude::PolarsResult;

/// Checks whether the geometries on the same row of two chunks of a `kind` column's storage
/// are equal, coordinate by coordinate.
/// Implements the Python `is_equal`.
///
/// `b` must be as long as `a`, or a single row.
pub fn are_equal(
    a: &dyn Array,
    b: &dyn Array,
    kind: Kind,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    nans_equal: bool,
) -> PolarsResult<BooleanArray> {
    let rows = |chunk| downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_are_equal(
            &PointArray::new(Coords::new(a)?),
            &PointArray::new(Coords::new(b)?),
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiPoint => multi_points_are_equal(
            &MultiPointArray::new_untrimmed(rows(a)?)?,
            &MultiPointArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::LineString => line_strings_are_equal(
            &LineStringArray::new_untrimmed(rows(a)?)?,
            &LineStringArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiLineString => multi_line_strings_are_equal(
            &MultiLineStringArray::new_untrimmed(rows(a)?)?,
            &MultiLineStringArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::Polygon => polygons_are_equal(
            &PolygonArray::new_untrimmed(rows(a)?)?,
            &PolygonArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiPolygon => multi_polygons_are_equal(
            &MultiPolygonArray::new_untrimmed(rows(a)?)?,
            &MultiPolygonArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
    })
}
