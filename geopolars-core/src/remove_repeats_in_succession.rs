use geopolars_compute::native::{
    line_strings_without_repeats_in_succession, multi_line_strings_without_repeats_in_succession,
    multi_points_without_repeats_in_succession, multi_polygons_without_repeats_in_succession,
    polygons_without_repeats_in_succession, Coincidence, LineOptions,
};
use geopolars_views::{
    downcast, Kind, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PolygonArray,
};
use polars_arrow::array::{Array, ListArray};
use polars_core::prelude::PolarsResult;

/// Removes the coordinates that coincide with the one before them,
/// from every geometry in one chunk of a `kind` column's storage.
/// Implements the Python `remove_repeats_in_succession`.
/// The result is the storage of the same kind and dimension.
pub fn without_repeats_in_succession(
    chunk: &dyn Array,
    kind: Kind,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    options: LineOptions,
) -> PolarsResult<Box<dyn Array>> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => chunk.to_boxed(),
        Kind::LineString => line_strings_without_repeats_in_succession(
            &LineStringArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
            options,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiPoint => multi_points_without_repeats_in_succession(
            &MultiPointArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        )
        .into_list_array()
        .boxed(),
        Kind::Polygon => polygons_without_repeats_in_succession(
            &PolygonArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
            options.too_short,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiLineString => multi_line_strings_without_repeats_in_succession(
            &MultiLineStringArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
            options,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiPolygon => multi_polygons_without_repeats_in_succession(
            &MultiPolygonArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
            options.too_short,
        )
        .into_list_array()
        .boxed(),
    })
}
