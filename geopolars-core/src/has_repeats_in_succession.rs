use geopolars_compute::native::{
    line_strings_have_repeats_in_succession, multi_line_strings_have_repeats_in_succession,
    multi_points_have_repeats_in_succession, multi_polygons_have_repeats_in_succession,
    points_have_repeats_in_succession, polygons_have_repeats_in_succession, Coincidence,
};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_core::prelude::PolarsResult;

/// Checks every geometry in one chunk of a `kind` column's storage
/// for two coordinates in succession that coincide.
/// Implements the Python `has_repeats_in_succession`.
pub fn have_repeats_in_succession(
    chunk: &dyn Array,
    kind: Kind,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
) -> PolarsResult<BooleanArray> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_have_repeats_in_succession(&PointArray::new(Coords::new(chunk)?)),
        Kind::LineString => line_strings_have_repeats_in_succession(
            &LineStringArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        ),
        Kind::MultiPoint => multi_points_have_repeats_in_succession(
            &MultiPointArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        ),
        Kind::MultiLineString => multi_line_strings_have_repeats_in_succession(
            &MultiLineStringArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        ),
        Kind::Polygon => polygons_have_repeats_in_succession(
            &PolygonArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        ),
        Kind::MultiPolygon => multi_polygons_have_repeats_in_succession(
            &MultiPolygonArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        ),
    })
}
