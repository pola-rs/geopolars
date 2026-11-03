//! Moving every null value below a row out to the row,
//! so that the views hold to GeoArrow's rule on null values:
//! nulls only at the outermost level.
//! This is a separate pass, before any kernel reads the views.

use geopolars_core::{
    BoxArray, LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray, UncheckedBoxes, UncheckedStorage, ValidCoords, ValidList,
};
use polars_arrow::bitmap::Bitmap;

use crate::passing::rows_passing;

/// The validity of the rows of `storage`, with every null value below a row moved out to it:
/// a row is also null where a part or coordinate (or one of its axes) below it is.
/// `None` when every row is there.
fn propagate_inner_nulls(storage: &UncheckedStorage) -> Option<Bitmap> {
    let lists: Vec<&ValidList> = storage.lists.iter().map(List::ignoring_validity).collect();
    let validities = (storage.lists.iter().map(List::validity)).chain([storage.coords.validity()]);
    rows_passing(
        &lists,
        validities.map(|validity| validity.cloned()).collect(),
    )
}

/// The `N` list levels of `storage`, the rows with the validity of [propagate_inner_nulls],
/// and its coordinates.
/// Storage with another number of list levels is a bug.
fn propagate_inner_nulls_of_lists<const N: usize>(
    storage: UncheckedStorage,
) -> ([List; N], ValidCoords) {
    let rows = propagate_inner_nulls(&storage);
    let coords = storage.coords.ignoring_validity().clone();
    let levels = storage.lists.len();
    let mut lists: [List; N] = storage
        .lists
        .try_into()
        .unwrap_or_else(|_| panic!("expected {N} list levels, got {levels}"));
    lists[0] = lists[0].clone().with_validity(rows);
    (lists, coords)
}

/// The points of `storage`, null where one of their axes is.
/// The other `propagate_inner_nulls_of_*` functions do the same for their kind:
/// a row is null where anything below it is.
pub fn propagate_inner_nulls_of_points(storage: UncheckedStorage) -> PointArray {
    let levels = storage.lists.len();
    assert_eq!(levels, 0, "expected points, got {levels} list levels");
    // The coordinates already combine the validity of the struct and its axes.
    PointArray::new(storage.coords)
}

pub fn propagate_inner_nulls_of_multi_points(storage: UncheckedStorage) -> MultiPointArray {
    let ([rows], coords) = propagate_inner_nulls_of_lists(storage);
    MultiPointArray::from_parts(rows, coords)
}

pub fn propagate_inner_nulls_of_line_strings(storage: UncheckedStorage) -> LineStringArray {
    let ([rows], coords) = propagate_inner_nulls_of_lists(storage);
    LineStringArray::from_parts(rows, coords)
}

pub fn propagate_inner_nulls_of_multi_line_strings(
    storage: UncheckedStorage,
) -> MultiLineStringArray {
    let ([rows, lines], coords) = propagate_inner_nulls_of_lists(storage);
    MultiLineStringArray::from_parts(rows, lines.ignoring_validity().clone(), coords)
}

pub fn propagate_inner_nulls_of_polygons(storage: UncheckedStorage) -> PolygonArray {
    let ([rows, rings], coords) = propagate_inner_nulls_of_lists(storage);
    PolygonArray::from_parts(rows, rings.ignoring_validity().clone(), coords)
}

pub fn propagate_inner_nulls_of_multi_polygons(storage: UncheckedStorage) -> MultiPolygonArray {
    let ([rows, polygons, rings], coords) = propagate_inner_nulls_of_lists(storage);
    MultiPolygonArray::from_parts(
        rows,
        polygons.ignoring_validity().clone(),
        rings.ignoring_validity().clone(),
        coords,
    )
}

/// The boxes already combine the validity of the struct and its bounds,
/// as the coordinates do in [propagate_inner_nulls_of_points].
pub fn propagate_inner_nulls_of_boxes(storage: UncheckedBoxes) -> BoxArray {
    storage.boxes
}
