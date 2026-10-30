//! Snapping every coordinate onto a grid, for `snap_to_grid`.
//! See `snap_to_grid` in Python for the rules.
//!
//! Every kernel only rewrites x, y and z, and keeps every offset.
//!
//! # Const generic flags
//!
//! ## `MODE`
//!
//! [snapped_axis] is generic over `MODE`, the discriminant of a [RoundMode],
//! as an enum cannot be a const generic.

use geopolars_core::{
    Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, ValidCoords,
};

/// Which grid line a coordinate goes to, as `mode` of Polars' `Expr.round`,
/// but counted in spacings from the origin.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum RoundMode {
    /// The nearest one. A tie goes to the one an even number of spacings from the origin.
    HalfToEven,
    /// The nearest one. A tie goes to the one further from the origin.
    HalfAwayFromZero,
    /// The one between the coordinate and the origin.
    ToZero,
}

const HALF_TO_EVEN: u8 = RoundMode::HalfToEven as u8;
const HALF_AWAY_FROM_ZERO: u8 = RoundMode::HalfAwayFromZero as u8;
const TO_ZERO: u8 = RoundMode::ToZero as u8;

/// A grid of lines `spacing` apart along each axis (x, y and z), one of them running through `origin`.
/// A spacing of 0 leaves that axis as it is.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    pub spacing: [f64; 3],
    pub origin: [f64; 3],
    pub mode: RoundMode,
}

fn snapped_axis<const MODE: u8>(values: &[f64], spacing: f64, origin: f64) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            let steps = (value - origin) / spacing;
            let steps = match MODE {
                HALF_TO_EVEN => steps.round_ties_even(),
                HALF_AWAY_FROM_ZERO => steps.round(),
                _ => steps.trunc(),
            };
            steps * spacing + origin
        })
        .collect()
}

/// `coords`, with x, y and z snapped onto `grid`. NaN stays NaN; m is shared.
fn snapped(coords: &ValidCoords, grid: &Grid) -> ValidCoords {
    let snap = |axis: usize, values: &[f64]| {
        let (spacing, origin) = (grid.spacing[axis], grid.origin[axis]);
        (spacing != 0.0).then(|| match grid.mode {
            RoundMode::HalfToEven => snapped_axis::<HALF_TO_EVEN>(values, spacing, origin),
            RoundMode::HalfAwayFromZero => {
                snapped_axis::<HALF_AWAY_FROM_ZERO>(values, spacing, origin)
            }
            RoundMode::ToZero => snapped_axis::<TO_ZERO>(values, spacing, origin),
        })
    };
    coords.with_positions([
        snap(0, coords.x()),
        snap(1, coords.y()),
        coords.z().and_then(|z| snap(2, z)),
    ])
}

pub fn points_snapped_to_grid(array: &PointArray, grid: &Grid) -> PointArray {
    PointArray::new(Coords::from_valid(
        snapped(array.coords.ignoring_validity(), grid),
        array.coords.validity().cloned(),
    ))
}

pub fn line_strings_snapped_to_grid(array: &LineStringArray, grid: &Grid) -> LineStringArray {
    LineStringArray::from_parts(array.rows.clone(), snapped(&array.coords, grid))
}

pub fn multi_points_snapped_to_grid(array: &MultiPointArray, grid: &Grid) -> MultiPointArray {
    MultiPointArray::from_parts(array.rows.clone(), snapped(&array.coords, grid))
}

pub fn polygons_snapped_to_grid(array: &PolygonArray, grid: &Grid) -> PolygonArray {
    PolygonArray::from_parts(
        array.rows.clone(),
        array.rings.clone(),
        snapped(&array.coords, grid),
    )
}

pub fn multi_line_strings_snapped_to_grid(
    array: &MultiLineStringArray,
    grid: &Grid,
) -> MultiLineStringArray {
    MultiLineStringArray::from_parts(
        array.rows.clone(),
        array.lines.clone(),
        snapped(&array.coords, grid),
    )
}

pub fn multi_polygons_snapped_to_grid(array: &MultiPolygonArray, grid: &Grid) -> MultiPolygonArray {
    MultiPolygonArray::from_parts(
        array.rows.clone(),
        array.polygons.clone(),
        array.rings.clone(),
        snapped(&array.coords, grid),
    )
}
