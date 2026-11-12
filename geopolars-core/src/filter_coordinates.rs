use geopolars_compute::native::{
    line_strings_with_selected, multi_line_strings_with_selected, multi_points_with_selected,
    multi_polygons_with_selected, points_with_selected, polygons_with_selected,
    CoordinateSelection, SelectedAction, TooShort,
};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::{Array, ListArray};
use polars_core::prelude::PolarsResult;

/// Takes `action` on the coordinates `selection` selects,
/// in every geometry in one chunk of a `kind` column's storage.
/// Implements the Python `filter_coordinates`, and `fix_out_of_bounds` unless it clamps.
/// The result is the storage of the same kind and dimension.
pub fn with_selected_coords(
    chunk: &dyn Array,
    kind: Kind,
    selection: &CoordinateSelection,
    action: SelectedAction,
    too_short: TooShort,
) -> PolarsResult<Box<dyn Array>> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => {
            points_with_selected(&PointArray::new(Coords::new(chunk)?), selection, action)
                .into_struct_array()
                .boxed()
        }
        Kind::LineString => line_strings_with_selected(
            &LineStringArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiPoint => multi_points_with_selected(
            &MultiPointArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )
        .into_list_array()
        .boxed(),
        Kind::Polygon => {
            polygons_with_selected(&PolygonArray::new(rows()?)?, selection, action, too_short)
                .into_list_array()
                .boxed()
        }
        Kind::MultiLineString => multi_line_strings_with_selected(
            &MultiLineStringArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )
        .into_list_array()
        .boxed(),
        Kind::MultiPolygon => multi_polygons_with_selected(
            &MultiPolygonArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )
        .into_list_array()
        .boxed(),
    })
}
