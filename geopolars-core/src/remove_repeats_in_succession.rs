use geopolars_compute::native::{
    line_strings_without_repeats_in_succession, multi_line_strings_without_repeats_in_succession,
    multi_points_without_repeats_in_succession, multi_polygons_without_repeats_in_succession,
    polygons_without_repeats_in_succession, Coincidence, LineOptions,
};
use geopolars_views::{
    downcast, Coords, GeometryArray, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::{Array, ListArray};
use polars_core::prelude::PolarsResult;

/// Removes the coordinates that coincide with the one before them,
/// from every geometry in one chunk of a `kind` column's storage.
/// Implements the Python `remove_repeats_in_succession`.
/// The result is of the same kind and dimension.
pub fn without_repeats_in_succession(
    chunk: &dyn Array,
    kind: Kind,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    options: LineOptions,
) -> PolarsResult<GeometryArray> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => GeometryArray::Point(PointArray::new(Coords::new(chunk)?)),
        Kind::LineString => GeometryArray::LineString(line_strings_without_repeats_in_succession(
            &LineStringArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
            options,
        )),
        Kind::MultiPoint => GeometryArray::MultiPoint(multi_points_without_repeats_in_succession(
            &MultiPointArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        )),
        Kind::Polygon => GeometryArray::Polygon(polygons_without_repeats_in_succession(
            &PolygonArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
            options.too_short,
        )),
        Kind::MultiLineString => {
            GeometryArray::MultiLineString(multi_line_strings_without_repeats_in_succession(
                &MultiLineStringArray::new(rows()?)?,
                coincidence,
                unique_z,
                unique_m,
                options,
            ))
        }
        Kind::MultiPolygon => {
            GeometryArray::MultiPolygon(multi_polygons_without_repeats_in_succession(
                &MultiPolygonArray::new(rows()?)?,
                coincidence,
                unique_z,
                unique_m,
                options.too_short,
            ))
        }
    })
}
