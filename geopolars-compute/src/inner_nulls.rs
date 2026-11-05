//! Moving every null value below a row out to the row,
//! so that the views hold to GeoArrow's rule on null values:
//! nulls only at the outermost level.
//! Of a multi-geometry, the parts with a null value can be left out instead.
//! This is a separate pass, before any kernel reads the views.

use geopolars_core::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, UncheckedStorage, ValidCoords, ValidList,
};
use polars_arrow::bitmap::Bitmap;

#[cfg(feature = "native")]
use crate::native::{
    multi_line_strings_with_kept_parts, multi_points_with_kept_parts,
    multi_polygons_with_kept_parts,
};
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
    let (mut lists, coords) = levels_of(storage);
    lists[0] = lists[0].clone().with_validity(rows);
    (lists, coords)
}

/// The `N` list levels of `storage` as they are, and its coordinates.
/// Storage with another number of list levels is a bug.
fn levels_of<const N: usize>(storage: UncheckedStorage) -> ([List; N], ValidCoords) {
    let coords = storage.coords.ignoring_validity().clone();
    let levels = storage.lists.len();
    let lists: [List; N] = storage
        .lists
        .try_into()
        .unwrap_or_else(|_| panic!("expected {N} list levels, got {levels}"));
    (lists, coords)
}

/// Which parts of the rows of `storage` have no null value in or below them.
/// `None` when every part does.
#[cfg(feature = "native")]
fn parts_passing(storage: &UncheckedStorage) -> Option<Bitmap> {
    let below_parts = &storage.lists[1..];
    let lists: Vec<&ValidList> = below_parts.iter().map(List::ignoring_validity).collect();
    let validities = (below_parts.iter().map(List::validity)).chain([storage.coords.validity()]);
    rows_passing(
        &lists,
        validities.map(|validity| validity.cloned()).collect(),
    )
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

/// The multipoints of `storage`, without the points that are null.
/// Unlike [propagate_inner_nulls_of_multi_points], only a null row is null.
/// The other `drop_null_parts_of_*` functions do the same for their kind:
/// they leave out every part with a null value in or below it.
#[cfg(feature = "native")]
pub fn drop_null_parts_of_multi_points(storage: UncheckedStorage) -> MultiPointArray {
    let passing = parts_passing(&storage);
    let ([rows], coords) = levels_of(storage);
    let array = MultiPointArray::from_parts(rows, coords);
    match passing {
        Some(passing) => multi_points_with_kept_parts(array, &passing),
        None => array,
    }
}

#[cfg(feature = "native")]
pub fn drop_null_parts_of_multi_line_strings(storage: UncheckedStorage) -> MultiLineStringArray {
    let passing = parts_passing(&storage);
    let ([rows, lines], coords) = levels_of(storage);
    let array = MultiLineStringArray::from_parts(rows, lines.ignoring_validity().clone(), coords);
    match passing {
        Some(passing) => multi_line_strings_with_kept_parts(array, &passing),
        None => array,
    }
}

#[cfg(feature = "native")]
pub fn drop_null_parts_of_multi_polygons(storage: UncheckedStorage) -> MultiPolygonArray {
    let passing = parts_passing(&storage);
    let ([rows, polygons, rings], coords) = levels_of(storage);
    let array = MultiPolygonArray::from_parts(
        rows,
        polygons.ignoring_validity().clone(),
        rings.ignoring_validity().clone(),
        coords,
    );
    match passing {
        Some(passing) => multi_polygons_with_kept_parts(array, &passing),
        None => array,
    }
}
