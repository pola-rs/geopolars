use geopolars_compute::native::{
    line_strings_with_selected, multi_line_strings_with_selected, multi_points_with_selected,
    multi_polygons_with_selected, points_with_selected, polygons_with_selected,
    CoordinateSelection, SelectedAction, TooShort,
};
use geopolars_views::{
    downcast, Coords, GeometryArray, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::{Array, ListArray};
use polars_core::prelude::PolarsResult;

/// Takes `action` on the coordinates `selection` selects,
/// in every geometry in one chunk of a `kind` column's storage.
/// Implements the Python `filter_coordinates`, and `fix_out_of_bounds` unless it clamps.
/// The result is of the same kind and dimension.
pub fn with_selected_coords(
    chunk: &dyn Array,
    kind: Kind,
    selection: &CoordinateSelection,
    action: SelectedAction,
    too_short: TooShort,
) -> PolarsResult<GeometryArray> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => GeometryArray::Point(points_with_selected(
            &PointArray::new(Coords::new(chunk)?),
            selection,
            action,
        )),
        Kind::LineString => GeometryArray::LineString(line_strings_with_selected(
            &LineStringArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )),
        Kind::MultiPoint => GeometryArray::MultiPoint(multi_points_with_selected(
            &MultiPointArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )),
        Kind::Polygon => GeometryArray::Polygon(polygons_with_selected(
            &PolygonArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )),
        Kind::MultiLineString => GeometryArray::MultiLineString(multi_line_strings_with_selected(
            &MultiLineStringArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )),
        Kind::MultiPolygon => GeometryArray::MultiPolygon(multi_polygons_with_selected(
            &MultiPolygonArray::new(rows()?)?,
            selection,
            action,
            too_short,
        )),
    })
}
