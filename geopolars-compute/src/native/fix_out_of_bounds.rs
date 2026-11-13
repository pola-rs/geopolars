//! Clamping the coordinates outside the CRS's bounds.
//! Implements the Python `fix_out_of_bounds` with `action="clamp"`.
//!
//! Every kernel only rewrites x and y, and keeps every offset.
//! The other actions select the coordinates outside the bounds
//! ([CoordinateSelection::bounds](super::CoordinateSelection::bounds)),
//! then act on them as in [super::selected_coords](mod@super::selected_coords).

use geopolars_views::{
    Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, ValidCoords,
};

use super::coordinate_selection::Bounds;
use super::selected_coords::SelectedAction;

/// See `action` of `OutOfBounds` in Python.
#[derive(Debug, Clone, Copy)]
pub enum OutOfBoundsAction {
    SetEmpty,
    Drop,
    SetNull,
    Clamp,
}

impl OutOfBoundsAction {
    /// The action on the coordinates outside the bounds, `None` for [OutOfBoundsAction::Clamp].
    pub fn selected_action(self) -> Option<SelectedAction> {
        match self {
            OutOfBoundsAction::SetEmpty => Some(SelectedAction::SetEmpty),
            OutOfBoundsAction::Drop => Some(SelectedAction::Drop),
            OutOfBoundsAction::SetNull => Some(SelectedAction::SetNull),
            OutOfBoundsAction::Clamp => None,
        }
    }
}

/// `coords`, with x and y clamped into `bounds`, sharing z and m.
fn clamped(coords: &ValidCoords, bounds: &Bounds) -> ValidCoords {
    let [xmin, ymin, xmax, ymax] = *bounds;
    let clamp = |axis: &[f64], min: f64, max: f64| -> Vec<f64> {
        axis.iter().map(|value| value.clamp(min, max)).collect()
    };
    coords.with_positions([
        Some(clamp(coords.x(), xmin, xmax)),
        Some(clamp(coords.y(), ymin, ymax)),
        None,
    ])
}

pub fn points_with_clamped_coords(array: &PointArray, bounds: &Bounds) -> PointArray {
    PointArray::new(Coords::from_valid(
        clamped(array.coords.ignoring_validity(), bounds),
        array.coords.validity().cloned(),
    ))
}

pub fn line_strings_with_clamped_coords(
    array: &LineStringArray,
    bounds: &Bounds,
) -> LineStringArray {
    LineStringArray::from_parts(array.rows.clone(), clamped(&array.coords, bounds))
}

pub fn multi_points_with_clamped_coords(
    array: &MultiPointArray,
    bounds: &Bounds,
) -> MultiPointArray {
    MultiPointArray::from_parts(array.rows.clone(), clamped(&array.coords, bounds))
}

pub fn polygons_with_clamped_coords(array: &PolygonArray, bounds: &Bounds) -> PolygonArray {
    PolygonArray::from_parts(
        array.rows.clone(),
        array.rings.clone(),
        clamped(&array.coords, bounds),
    )
}

pub fn multi_line_strings_with_clamped_coords(
    array: &MultiLineStringArray,
    bounds: &Bounds,
) -> MultiLineStringArray {
    MultiLineStringArray::from_parts(
        array.rows.clone(),
        array.lines.clone(),
        clamped(&array.coords, bounds),
    )
}

pub fn multi_polygons_with_clamped_coords(
    array: &MultiPolygonArray,
    bounds: &Bounds,
) -> MultiPolygonArray {
    MultiPolygonArray::from_parts(
        array.rows.clone(),
        array.polygons.clone(),
        array.rings.clone(),
        clamped(&array.coords, bounds),
    )
}
