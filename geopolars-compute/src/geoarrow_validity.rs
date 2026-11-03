//! Whether each geometry is valid by the GeoArrow spec, and holds no selected coordinate.
//! See `is_valid_geoarrow` in Python for the rules.
//!
//! Every check writes a mask of the items that pass at its own level.
//! These levels are the coordinates (`which_are_not_selected`) and the rings (`rings_are_closed`).
//! A box only has its ranges (`ranges_are_finite_or_empty`).
//! `rows_passing` folds the masks up to the rows.
//! Each kernel gives a `bool` per row, `false` for a null row.
//!
//! Null values below the rows are not looked at here:
//! the views have to be built with them moved out to the rows (see [crate::inner_nulls]).

use geopolars_core::{
    BoxArray, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;

use crate::native::{rings_are_closed, which_are_not_selected, CoordinateSelection};
use crate::passing::{passing_bool_array, rows_passing};

pub fn points_are_valid_geoarrow(
    array: &PointArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let coords = &array.coords;
    let unselected = which_are_not_selected(coords.ignoring_validity(), selection);
    let passing = combine_validities_and(coords.validity(), unselected.as_ref());
    let len = coords.ignoring_validity().len();
    passing_bool_array(len, coords.validity(), passing)
}

pub fn multi_points_are_valid_geoarrow(
    array: &MultiPointArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let rows = &array.rows;
    let passing = rows_passing(
        &[rows.ignoring_validity()],
        vec![
            rows.validity().cloned(),
            which_are_not_selected(&array.coords, selection),
        ],
    );
    passing_bool_array(rows.len(), rows.validity(), passing)
}

pub fn line_strings_are_valid_geoarrow(
    array: &LineStringArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let rows = &array.rows;
    let passing = rows_passing(
        &[rows.ignoring_validity()],
        vec![
            rows.validity().cloned(),
            which_are_not_selected(&array.coords, selection),
        ],
    );
    passing_bool_array(rows.len(), rows.validity(), passing)
}

pub fn multi_line_strings_are_valid_geoarrow(
    array: &MultiLineStringArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let rows = &array.rows;
    let passing = rows_passing(
        &[rows.ignoring_validity(), &array.lines],
        vec![
            rows.validity().cloned(),
            None,
            which_are_not_selected(&array.coords, selection),
        ],
    );
    passing_bool_array(rows.len(), rows.validity(), passing)
}

pub fn polygons_are_valid_geoarrow(
    array: &PolygonArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let rows = &array.rows;
    let passing = rows_passing(
        &[rows.ignoring_validity(), &array.rings],
        vec![
            rows.validity().cloned(),
            Some(rings_are_closed(&array.rings, &array.coords)),
            which_are_not_selected(&array.coords, selection),
        ],
    );
    passing_bool_array(rows.len(), rows.validity(), passing)
}

pub fn multi_polygons_are_valid_geoarrow(
    array: &MultiPolygonArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let rows = &array.rows;
    let passing = rows_passing(
        &[rows.ignoring_validity(), &array.polygons, &array.rings],
        vec![
            rows.validity().cloned(),
            None,
            Some(rings_are_closed(&array.rings, &array.coords)),
            which_are_not_selected(&array.coords, selection),
        ],
    );
    passing_bool_array(rows.len(), rows.validity(), passing)
}

pub fn boxes_are_valid_geoarrow(array: &BoxArray, check_finite_coordinates: bool) -> BooleanArray {
    let finite = check_finite_coordinates.then(|| ranges_are_finite_or_empty(array));
    let passing = combine_validities_and(array.validity(), finite.as_ref());
    passing_bool_array(array.len(), array.validity(), passing)
}

fn ranges_are_finite_or_empty(array: &BoxArray) -> Bitmap {
    (0..array.axes())
        .map(|axis| {
            let ranges = array.min(axis).iter().zip(array.max(axis));
            Bitmap::from_trusted_len_iter(
                ranges.map(|(&min, &max)| range_is_finite_or_empty(min, max)),
            )
        })
        .reduce(|passing, axis| &passing & &axis)
        .unwrap()
}

/// The spec writes an empty range as `inf` to `-inf`.
#[inline(always)]
fn range_is_finite_or_empty(min: f64, max: f64) -> bool {
    (min.is_finite() & max.is_finite()) | ((min == f64::INFINITY) & (max == f64::NEG_INFINITY))
}
