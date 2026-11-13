//! The bounding box of a geometry,
//! as an [Extent] over its coordinates that [BoxOptions::finish] turns into a box.
//! Implements the Python `bounds` and `total_bounds`.
//!
//! Only the coordinates matter, but every kind still gets its own wrapper, for consistency.
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! Every function here is generic over `Z`.
//! If `Z=true`, z is bounded too.
//!
//! ## `M`
//!
//! Every function here is generic over `M`.
//! If `M=true`, m is bounded too.
//!
//! ## `IGNORE_NULLS`
//!
//! The `*_column_extent` functions are generic over `IGNORE_NULLS`.
//! If `IGNORE_NULLS=false`, a null row makes the column's box null.
//! If `IGNORE_NULLS=true`, a null row is skipped.

use std::convert::Infallible;
use std::ops::Range;

use geopolars_views::output::{box_struct, list_array};
use geopolars_views::{
    GeoDimension, LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray, Untrimmed, ValidCoords,
};
use polars_arrow::array::{ListArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::Offsets;

use crate::per_row::per_row_values;

/// The minimum and maximum of every axis over the coordinates added so far.
/// An axis without any (non-NaN) value ranges from `inf` to `-inf`.
#[derive(Clone, Copy)]
pub struct Extent<const Z: bool, const M: bool> {
    /// Per axis, in storage order: x, y, then z and/or m.
    /// An axis that is not there stays `inf`.
    pub min: [f64; 4],
    /// See `min`. An axis that is not there stays `-inf`.
    pub max: [f64; 4],
}

impl<const Z: bool, const M: bool> Extent<Z, M> {
    const AXES: usize = 2 + Z as usize + M as usize;

    pub const EMPTY: Self = Self {
        min: [f64::INFINITY; 4],
        max: [f64::NEG_INFINITY; 4],
    };

    #[inline(always)]
    pub fn add_range(&mut self, coords: &ValidCoords, range: Range<usize>) {
        for k in 0..Self::AXES {
            let values = &coords.axis(k)[range.clone()];
            // `f64::min` and `f64::max` skip NaN: an empty point, or a lone NaN value.
            self.min[k] = values.iter().fold(self.min[k], |low, &v| low.min(v));
            self.max[k] = values.iter().fold(self.max[k], |high, &v| high.max(v));
        }
    }

    #[inline(always)]
    pub fn merge(&mut self, other: &Self) {
        for k in 0..Self::AXES {
            self.min[k] = self.min[k].min(other.min[k]);
            self.max[k] = self.max[k].max(other.max[k]);
        }
    }
}

impl<const Z: bool, const M: bool> Default for Extent<Z, M> {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// How an [Extent] becomes a box, as the Python `bounds` describes.
#[derive(Clone, Copy, Debug)]
pub struct BoxOptions {
    pub margin_x: f64,
    pub margin_y: f64,
    /// How far x goes in one full turn around the globe, in the CRS's unit.
    /// `None` leaves x as it is.
    pub turn: Option<f64>,
}

impl BoxOptions {
    /// Turns `extent` into a box.
    /// It is widened by the margins, with x brought back around the globe if there is a `turn`.
    #[inline(always)]
    pub fn finish<const Z: bool, const M: bool>(&self, extent: &Extent<Z, M>) -> Extent<Z, M> {
        let mut bounds = *extent;
        let has_x = bounds.min[0] <= bounds.max[0];
        for (k, margin) in [self.margin_x, self.margin_y].into_iter().enumerate() {
            bounds.min[k] -= margin;
            bounds.max[k] += margin;
        }
        if let (Some(turn), true) = (self.turn, has_x) {
            // After the margin, which can push a box over the antimeridian too.
            (bounds.min[0], bounds.max[0]) = around_the_globe(bounds.min[0], bounds.max[0], turn);
        }
        for k in 0..Extent::<Z, M>::AXES {
            if bounds.min[k].is_nan() {
                bounds.min[k] = f64::INFINITY;
            }
            if bounds.max[k].is_nan() {
                bounds.max[k] = f64::NEG_INFINITY;
            }
        }
        bounds
    }
}

/// Brings the longitude range `low..high` back to start in `[-turn / 2, turn / 2)`
#[inline(always)]
fn around_the_globe(low: f64, high: f64, turn: f64) -> (f64, f64) {
    let start = -turn / 2.0;
    if high - low >= turn {
        return (start, -start);
    }
    // Both ends move by the turns that bring `low` into range,
    // so a range lying entirely past the antimeridian moves as a whole.
    let turns = ((low - start) / turn).floor();
    let (low, high) = (low - turns * turn, high - turns * turn);
    (low, if high > -start { high - turn } else { high })
}

#[inline(always)]
fn bound<const Z: bool, const M: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
    options: &BoxOptions,
) -> Extent<Z, M> {
    let mut extent = Extent::EMPTY;
    extent.add_range(coords, vertices);
    options.finish(&extent)
}

/// The struct of a `geoarrow.box` column holding `boxes`.
pub fn box_array<const Z: bool, const M: bool>(
    dim: GeoDimension,
    boxes: &[Extent<Z, M>],
    validity: Option<Bitmap>,
) -> StructArray {
    let axes = 0..Extent::<Z, M>::AXES;
    let mins = axes
        .clone()
        .map(|k| boxes.iter().map(|b| b.min[k]).collect());
    let maxes = axes.map(|k| boxes.iter().map(|b| b.max[k]).collect());
    box_struct(dim, mins.chain(maxes).collect(), validity)
}

/// The box of every row of `rows`, whose coordinates sit at `vertices_of(i)`.
#[inline(always)]
fn row_boxes<const Z: bool, const M: bool>(
    rows: &List,
    coords: &ValidCoords,
    options: &BoxOptions,
    vertices_of: impl Fn(usize) -> Range<usize>,
) -> StructArray {
    let Ok(boxes) = per_row_values(rows, |i| {
        Ok::<_, Infallible>(bound::<Z, M>(coords, vertices_of(i), options))
    });
    box_array(coords.dim(), &boxes, rows.validity().cloned())
}

/// The box of every part of `rows`, as a list per row.
/// The coordinates of part `j` sit at `vertices_of(j)`.
#[inline(always)]
fn part_boxes<const Z: bool, const M: bool>(
    rows: &List,
    coords: &ValidCoords,
    options: &BoxOptions,
    vertices_of: impl Fn(usize) -> Range<usize>,
) -> ListArray<i64> {
    // Only the parts of these rows: a chunk can be a slice of a larger array.
    let boxes: Vec<Extent<Z, M>> = rows
        .span(0..rows.len())
        .map(|j| bound(coords, vertices_of(j), options))
        .collect();
    let offsets = rows.ignoring_validity().offsets().buffer();
    let first = offsets[0];
    let offsets = offsets.iter().map(|offset| offset - first).collect();
    // SAFETY: shifting every offset down by the first keeps them growing, from 0.
    let offsets = unsafe { Offsets::new_unchecked(offsets) }.into();
    let boxes = box_array(coords.dim(), &boxes, None);
    list_array(boxes.boxed(), offsets, rows.validity().cloned())
}

pub fn point_bounds<const Z: bool, const M: bool>(
    array: &PointArray,
    options: &BoxOptions,
) -> StructArray {
    let coords = array.coords.ignoring_validity();
    // A null point is bounded too: its box is masked by the validity.
    let boxes: Vec<Extent<Z, M>> = (0..coords.len())
        .map(|i| bound(coords, i..i + 1, options))
        .collect();
    box_array(coords.dim(), &boxes, array.coords.validity().cloned())
}

pub fn line_string_bounds<const Z: bool, const M: bool>(
    array: &Untrimmed<LineStringArray>,
    options: &BoxOptions,
) -> StructArray {
    row_boxes::<Z, M>(array.rows(), array.coords(), options, |i| {
        array.rows().range(i)
    })
}

pub fn polygon_bounds<const Z: bool, const M: bool>(
    array: &Untrimmed<PolygonArray>,
    options: &BoxOptions,
) -> StructArray {
    row_boxes::<Z, M>(array.rows(), array.coords(), options, |i| {
        array.rings().span(array.rows().range(i))
    })
}

/// One box per row, around all of its points.
pub fn multi_point_bounds<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiPointArray>,
    options: &BoxOptions,
) -> StructArray {
    row_boxes::<Z, M>(array.rows(), array.coords(), options, |i| {
        array.rows().range(i)
    })
}

/// One box per row, around all of its linestrings.
pub fn multi_line_string_bounds<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiLineStringArray>,
    options: &BoxOptions,
) -> StructArray {
    row_boxes::<Z, M>(array.rows(), array.coords(), options, |i| {
        array.lines().span(array.rows().range(i))
    })
}

/// One box per row, around all of its polygons.
pub fn multi_polygon_bounds<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiPolygonArray>,
    options: &BoxOptions,
) -> StructArray {
    row_boxes::<Z, M>(array.rows(), array.coords(), options, |i| {
        array
            .rings()
            .span(array.polygons().span(array.rows().range(i)))
    })
}

/// A list of boxes per row, one per point.
pub fn multi_point_part_bounds<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiPointArray>,
    options: &BoxOptions,
) -> ListArray<i64> {
    part_boxes::<Z, M>(array.rows(), array.coords(), options, |j| j..j + 1)
}

/// A list of boxes per row, one per linestring.
pub fn multi_line_string_part_bounds<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiLineStringArray>,
    options: &BoxOptions,
) -> ListArray<i64> {
    part_boxes::<Z, M>(array.rows(), array.coords(), options, |j| {
        array.lines().range(j)
    })
}

/// A list of boxes per row, one per polygon.
pub fn multi_polygon_part_bounds<const Z: bool, const M: bool>(
    array: &Untrimmed<MultiPolygonArray>,
    options: &BoxOptions,
) -> ListArray<i64> {
    part_boxes::<Z, M>(array.rows(), array.coords(), options, |j| {
        array.rings().span(array.polygons().range(j))
    })
}

/// The extent of every row of `rows`, before [BoxOptions::finish].
/// `vertices_of` maps a run of rows to where their coordinates sit.
/// `None` when the column's box is null.
#[inline(always)]
fn column_extent<const Z: bool, const M: bool, const IGNORE_NULLS: bool>(
    rows: &List,
    coords: &ValidCoords,
    vertices_of: impl Fn(Range<usize>) -> Range<usize>,
) -> Option<Extent<Z, M>> {
    let mut extent = Extent::EMPTY;
    if rows.is_all_valid() {
        extent.add_range(coords, vertices_of(0..rows.len()));
    } else if IGNORE_NULLS {
        for i in (0..rows.len()).filter(|&i| rows.is_valid(i)) {
            extent.add_range(coords, vertices_of(i..i + 1));
        }
    } else {
        return None;
    }
    Some(extent)
}

pub fn point_column_extent<const Z: bool, const M: bool, const IGNORE_NULLS: bool>(
    array: &PointArray,
) -> Option<Extent<Z, M>> {
    let coords = array.coords.ignoring_validity();
    let mut extent = Extent::EMPTY;
    if array.coords.is_all_valid() {
        extent.add_range(coords, 0..coords.len());
    } else if IGNORE_NULLS {
        for i in (0..coords.len()).filter(|&i| array.coords.is_valid(i)) {
            extent.add_range(coords, i..i + 1);
        }
    } else {
        return None;
    }
    Some(extent)
}

pub fn line_string_column_extent<const Z: bool, const M: bool, const IGNORE_NULLS: bool>(
    array: &LineStringArray,
) -> Option<Extent<Z, M>> {
    column_extent::<Z, M, IGNORE_NULLS>(&array.rows, &array.coords, |rows| array.rows.span(rows))
}

pub fn polygon_column_extent<const Z: bool, const M: bool, const IGNORE_NULLS: bool>(
    array: &PolygonArray,
) -> Option<Extent<Z, M>> {
    column_extent::<Z, M, IGNORE_NULLS>(&array.rows, &array.coords, |rows| {
        array.rings.span(array.rows.span(rows))
    })
}

pub fn multi_point_column_extent<const Z: bool, const M: bool, const IGNORE_NULLS: bool>(
    array: &MultiPointArray,
) -> Option<Extent<Z, M>> {
    column_extent::<Z, M, IGNORE_NULLS>(&array.rows, &array.coords, |rows| array.rows.span(rows))
}

pub fn multi_line_string_column_extent<const Z: bool, const M: bool, const IGNORE_NULLS: bool>(
    array: &MultiLineStringArray,
) -> Option<Extent<Z, M>> {
    column_extent::<Z, M, IGNORE_NULLS>(&array.rows, &array.coords, |rows| {
        array.lines.span(array.rows.span(rows))
    })
}

pub fn multi_polygon_column_extent<const Z: bool, const M: bool, const IGNORE_NULLS: bool>(
    array: &MultiPolygonArray,
) -> Option<Extent<Z, M>> {
    column_extent::<Z, M, IGNORE_NULLS>(&array.rows, &array.coords, |rows| {
        array.rings.span(array.polygons.span(array.rows.span(rows)))
    })
}
