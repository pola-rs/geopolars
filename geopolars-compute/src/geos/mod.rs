//! Interfacing with the `geos` crate.
//!
//! GEOS, as a C++ library, owns its coordinate storage,
//! so arrow buffers can't be borrowed zero-copy.
//! The copy is one memcpy-like op, so still not very heavy.
//!
//! # Const generic flags
//!
//! ## `EARLY_FINITE_CHECK`
//!
//! Converting rows to GEOS, and the convex hulls, are generic over `EARLY_FINITE_CHECK`.
//! [`ToGeosRows`] takes it as `early_finite_check`.
//! If `EARLY_FINITE_CHECK=true`, a row with a NaN or infinite x or y never goes to GEOS.
//! It is an error instead, which `ignore_per_row_errors` treats like a GEOS error.
//! For a point, the empty point (x and y NaN) is the exception.
//! If `EARLY_FINITE_CHECK=false`, those coordinates go to GEOS as they are.
//! See `early_finite_check` in Python for what GEOS makes of them.
//!
//! ## `ALL_VALID`
//!
//! The loops over the rows of a list array are generic over `ALL_VALID`.
//! It is set when no row is null ([`List::is_all_valid`]).
//! If `ALL_VALID=true`, every row is there, so none is checked for being null.
//!
//! ## `Z`
//!
//! Copying linestrings and polygons out of GEOS, and the convex hulls, are generic over `Z`.
//! If `Z=true`, the z values are copied too.
//!
//! ## `REVERSED`
//!
//! Copying a linestring or a ring out of GEOS is generic over `REVERSED`.
//! If `REVERSED=true`, its coordinates are copied in reverse order.
//! For a polygon, this flips its orientation.

mod buffer;
mod convex_hull;
mod interpolate;
mod intersection;
mod is_simple;
mod is_topologically_equal;
mod linestrings;
mod offset_curve;
mod pairs;
mod points;
mod polygons;
mod rings_overlap;
mod shortest_line;
mod simplify;
mod topology;

pub use buffer::{
    buffer_into_multi_polygons, buffer_into_polygons, BufferDistance, BufferStyle, CapStyle,
    JoinStyle, Side,
};
pub use convex_hull::{
    line_string_convex_hulls, multi_line_string_convex_hulls, multi_point_convex_hulls,
    multi_polygon_convex_hulls, polygon_convex_hulls,
};
pub use interpolate::{interpolate_line_strings, interpolate_multi_line_strings, InterpolateMode};
pub use intersection::IntersectionPairs;
pub use is_simple::{
    line_strings_are_simple, multi_line_strings_are_simple, multi_polygons_are_simple,
    polygons_are_simple,
};
pub use is_topologically_equal::topologically_equal;
pub use offset_curve::{
    offset_curves_into_line_strings, offset_curves_into_multi_line_strings, OffsetCurveStyle,
};
pub use rings_overlap::{rings_overlap_against_one, rings_overlap_pairwise};
pub use shortest_line::{shortest_lines_against_one, shortest_lines_pairwise};
pub use simplify::{
    simplify_line_strings, simplify_multi_line_strings, simplify_multi_polygons, simplify_polygons,
};
pub use topology::{meet_only_at_points_and_no_overlap, polygon_rings_fit};

use std::ops::Range;

use geopolars_core::output::SizeHint;
use geopolars_core::{
    Coords, GeoDimension, GeometryArray, List, MultiPointArray, PointArray, ScalarPoint,
    ValidCoords,
};
use geos::{CoordSeq, CoordType, Geom, Geometry, GeometryTypes};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::offset::{Offsets, OffsetsBuffer};

use crate::{row_or_null, GeoComputeError, GeoComputeResult};

/// Converts a single geometry into a GEOS geometry.
/// For an array, see [`ToGeosRows`].
pub trait ToGeos {
    type Geos;

    fn to_geos(&self) -> GeoComputeResult<Self::Geos>;
}

/// Converts an array into one [`GeosRow`] per row.
/// A null row is `Null`.
/// A row without parts is `Empty`, so it never goes to GEOS.
/// The size hint is the size of the array itself.
/// If `ignore_per_row_errors=true`, a row GEOS fails to build is null too;
/// otherwise its error is returned.
pub trait ToGeosRows {
    fn to_geos_rows(
        &self,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        self.to_geos_rows_taking_as_empty(None, ignore_per_row_errors, early_finite_check)
    }

    /// See [`ToGeosRows::to_geos_rows`].
    /// A row that is set in `as_empty` and not null is also `Empty`, without going to GEOS.
    fn to_geos_rows_taking_as_empty(
        &self,
        as_empty: Option<&Bitmap>,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries>;
}

impl ToGeosRows for GeometryArray {
    fn to_geos_rows_taking_as_empty(
        &self,
        as_empty: Option<&Bitmap>,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        let (ignore, early) = (ignore_per_row_errors, early_finite_check);
        match self {
            GeometryArray::Point(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, ignore, early)
            }
            GeometryArray::LineString(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, ignore, early)
            }
            GeometryArray::Polygon(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, ignore, early)
            }
            GeometryArray::MultiPoint(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, ignore, early)
            }
            GeometryArray::MultiLineString(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, ignore, early)
            }
            GeometryArray::MultiPolygon(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, ignore, early)
            }
        }
    }
}

fn coord_type(dim: GeoDimension) -> CoordType {
    match dim {
        GeoDimension::XY => CoordType::XY,
        GeoDimension::XYZ => CoordType::XYZ,
        GeoDimension::XYM => CoordType::XYM,
        GeoDimension::XYZM => CoordType::XYZM,
    }
}

/// `None` if the point is empty (null).
impl ToGeos for ScalarPoint<'_> {
    type Geos = Option<Geometry>;

    fn to_geos(&self) -> GeoComputeResult<Option<Geometry>> {
        let dim = self.coords.dim();
        point_to_geos::<false>(
            self.coords,
            self.index,
            dim.field_names().len(),
            coord_type(dim),
        )
    }
}

impl ToGeosRows for PointArray {
    fn to_geos_rows_taking_as_empty(
        &self,
        as_empty: Option<&Bitmap>,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        let geometries = with_const_flags!(
            EARLY_FINITE_CHECK = early_finite_check;
            points_to_geos::<EARLY_FINITE_CHECK>(&self.coords, as_empty, ignore_per_row_errors)
        )?;
        Ok(GeosGeometries {
            geometries,
            size_hint: SizeHint {
                coords: self.coords.len(),
                parts: [0, 0],
            },
        })
    }
}

fn points_to_geos<const EARLY_FINITE_CHECK: bool>(
    coords: &Coords,
    as_empty: Option<&Bitmap>,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<Vec<GeosRow>> {
    // GEOS has no batch constructor, so only the per-column work is shared.
    let dim = coords.dim();
    let len = dim.field_names().len();
    let coord_type = coord_type(dim);
    let mut geometries = Vec::with_capacity(coords.len());
    for i in 0..coords.len() {
        if is_taken_as_empty(as_empty, i) && coords.is_valid(i) {
            geometries.push(GeosRow::Empty);
            continue;
        }
        let point = point_to_geos::<EARLY_FINITE_CHECK>(coords, i, len, coord_type);
        geometries.push(row_or_null(point, ignore_per_row_errors)?.into());
    }
    Ok(geometries)
}

impl ToGeosRows for MultiPointArray {
    fn to_geos_rows_taking_as_empty(
        &self,
        as_empty: Option<&Bitmap>,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        let axes = axes(&self.coords);
        let size_hint = SizeHint {
            coords: self.coords.len(),
            parts: [0, 0],
        };
        let geometries = rows_to_geos(
            &self.rows,
            &axes,
            as_empty,
            ignore_per_row_errors,
            early_finite_check,
            |vertices| vertices,
            |vertices| multi_point_to_geos(&axes, vertices),
        )?;
        Ok(GeosGeometries {
            geometries,
            size_hint,
        })
    }
}

/// No vertices gives an empty multipoint.
#[inline(always)]
fn multi_point_to_geos(axes: &Axes, vertices: Range<usize>) -> GeoComputeResult<Geometry> {
    // GEOS takes ownership of the parts, so each point gets its own coordinates.
    let mut points = Vec::with_capacity(vertices.len());
    for j in vertices {
        points.push(Geometry::create_point(coord_seq(axes, j..j + 1)?)?);
    }
    Ok(Geometry::create_multipoint(points)?)
}

#[inline(always)]
fn is_geoarrow_empty_point(x: f64, y: f64) -> bool {
    x.is_nan() && y.is_nan()
}

/// Marks every [empty point](is_empty_point) and every row without parts in `array`.
/// A row whose parts are all empty, such as a polygon whose only ring has no vertices, is not marked.
/// The bit of a null row is arbitrary.
fn empty_rows(array: &GeometryArray) -> Bitmap {
    let without_parts =
        |rows: &List| Bitmap::from_iter((0..rows.len()).map(|i| rows.range(i).is_empty()));
    match array {
        GeometryArray::Point(array) => {
            let coords = array.coords.ignoring_validity();
            let (x, y) = (coords.x(), coords.y());
            Bitmap::from_iter(
                x.iter()
                    .zip(y)
                    .map(|(&x, &y)| is_geoarrow_empty_point(x, y)),
            )
        }
        GeometryArray::LineString(array) => without_parts(&array.rows),
        GeometryArray::Polygon(array) => without_parts(&array.rows),
        GeometryArray::MultiPoint(array) => without_parts(&array.rows),
        GeometryArray::MultiLineString(array) => without_parts(&array.rows),
        GeometryArray::MultiPolygon(array) => without_parts(&array.rows),
    }
}

/// Point `i` of `coords`, which carry `len` axes of `coord_type`.
/// `None` if it is null.
#[inline(always)]
fn point_to_geos<const EARLY_FINITE_CHECK: bool>(
    coords: &Coords,
    i: usize,
    len: usize,
    coord_type: CoordType,
) -> GeoComputeResult<Option<Geometry>> {
    let Some((x, y)) = coords.xy(i) else {
        return Ok(None);
    };
    if is_geoarrow_empty_point(x, y) {
        return Ok(Some(Geometry::create_empty_point()?));
    }
    if EARLY_FINITE_CHECK {
        xy_is_finite_or_error(&[x], &[y])?;
    }
    let mut buffer = [0.0; 4];
    for (n, value) in buffer[..len].iter_mut().enumerate() {
        *value = coords.nth(i, n);
    }
    let seq = CoordSeq::new_from_buffer(&buffer[..len], 1, coord_type)?;
    Ok(Some(Geometry::create_point(seq)?))
}

/// The rows of [`ToGeosRows`] for a list array.
/// `to_geos` builds one geometry from the range of a row's parts.
/// `span` gives where the coordinates of those parts sit in `axes`.
fn rows_to_geos(
    rows: &List,
    axes: &Axes,
    as_empty: Option<&Bitmap>,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
    span: impl Fn(Range<usize>) -> Range<usize>,
    to_geos: impl Fn(Range<usize>) -> GeoComputeResult<Geometry>,
) -> GeoComputeResult<Vec<GeosRow>> {
    with_const_flags!(
        ALL_VALID = rows.is_all_valid(), EARLY_FINITE_CHECK = early_finite_check;
        rows_to_geos_with::<ALL_VALID, EARLY_FINITE_CHECK>(
            rows,
            axes,
            as_empty,
            ignore_per_row_errors,
            span,
            to_geos,
        )
    )
}

/// See [`rows_to_geos`].
/// Errors are the exception, so `ignore_per_row_errors` is checked per row rather than made const.
fn rows_to_geos_with<const ALL_VALID: bool, const EARLY_FINITE_CHECK: bool>(
    rows: &List,
    axes: &Axes,
    as_empty: Option<&Bitmap>,
    ignore_per_row_errors: bool,
    span: impl Fn(Range<usize>) -> Range<usize>,
    to_geos: impl Fn(Range<usize>) -> GeoComputeResult<Geometry>,
) -> GeoComputeResult<Vec<GeosRow>> {
    let (x, y, _, _) = axes;
    let mut geometries = Vec::with_capacity(rows.len());
    for i in 0..rows.len() {
        // A null row's offsets point anywhere, so it is not converted.
        if !ALL_VALID && !rows.is_valid(i) {
            geometries.push(GeosRow::Null);
            continue;
        }
        let parts = rows.range(i);
        geometries.push(if parts.is_empty() || is_taken_as_empty(as_empty, i) {
            GeosRow::Empty
        } else {
            let geometry = if EARLY_FINITE_CHECK {
                let coords = span(parts.clone());
                xy_is_finite_or_error(&x[coords.clone()], &y[coords]).and_then(|()| to_geos(parts))
            } else {
                to_geos(parts)
            };
            row_or_null(geometry.map(Some), ignore_per_row_errors)?.into()
        });
    }
    Ok(geometries)
}

#[inline(always)]
fn is_taken_as_empty(as_empty: Option<&Bitmap>, i: usize) -> bool {
    as_empty.is_some_and(|as_empty| as_empty.get_bit(i))
}

/// The x, y, z and m values of every coordinate, as [`CoordSeq::new_from_arrays`] takes them.
type Axes<'a> = (&'a [f64], &'a [f64], Option<&'a [f64]>, Option<&'a [f64]>);

fn axes(coords: &ValidCoords) -> Axes<'_> {
    let (x, y) = (coords.axis(0), coords.axis(1));
    match coords.dim() {
        GeoDimension::XY => (x, y, None, None),
        GeoDimension::XYZ => (x, y, Some(coords.axis(2)), None),
        GeoDimension::XYM => (x, y, None, Some(coords.axis(2))),
        GeoDimension::XYZM => (x, y, Some(coords.axis(2)), Some(coords.axis(3))),
    }
}

/// An error for the first coordinate with a NaN or infinite `x` or `y`, if there is one.
/// GEOS takes neither, and has no empty point inside a geometry.
/// z is not checked: GEOS itself uses a NaN z for a coordinate without one.
#[inline(always)]
fn xy_is_finite_or_error(x: &[f64], y: &[f64]) -> GeoComputeResult<()> {
    // Does not stop at the first that fails, so that it vectorises.
    let finite = |axis: &[f64]| axis.iter().fold(true, |ok, value| ok & value.is_finite());
    if finite(x) && finite(y) {
        return Ok(());
    }
    let position = x
        .iter()
        .zip(y)
        .map(|(&x, &y)| (x, y))
        .find(|(x, y)| !(x.is_finite() && y.is_finite()))
        .expect("a non-finite coordinate, as the check above failed");
    Err(GeoComputeError::NotFinite { position })
}

/// The coordinates at `vertices`, copied from `axes` in one go.
#[inline(always)]
fn coord_seq(axes: &Axes, vertices: Range<usize>) -> GeoComputeResult<CoordSeq> {
    let (x, y, z, m) = axes;
    Ok(CoordSeq::new_from_arrays(
        &x[vertices.clone()],
        &y[vertices.clone()],
        z.map(|z| &z[vertices.clone()]),
        m.map(|m| &m[vertices]),
    )?)
}

/// GEOS geometries of any kind, one per row,
/// to copy into one of our arrays with an `into_*_array` conversion.
pub struct GeosGeometries {
    pub geometries: Vec<GeosRow>,
    /// Sized for the deepest kind, a multipolygon:
    /// rings (or lines), then polygons.
    pub size_hint: SizeHint<2>,
}

/// One row of [`GeosGeometries`].
pub enum GeosRow {
    Null,
    /// Empty, without a GEOS geometry. Valid for every kind.
    Empty,
    Geometry(Geometry),
}

/// `None` is a null row.
impl From<Option<Geometry>> for GeosRow {
    fn from(geometry: Option<Geometry>) -> Self {
        geometry.map_or(GeosRow::Null, GeosRow::Geometry)
    }
}

/// For a conversion getting a geometry of a kind it can't hold:
/// `false` (its row is null) if `ignore_per_row_errors=true`, otherwise an error.
fn wrong_kind(
    expected: &str,
    found: GeometryTypes,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<bool> {
    if ignore_per_row_errors {
        return Ok(false);
    }
    Err(geos::Error::ConversionError(format!("expected a {expected}, got a {found:?}")).into())
}

/// `geometry`, unless it is a `multi` of several parts, which the single kind cannot hold.
fn unless_split(geometry: Geometry, multi: GeometryTypes) -> GeoComputeResult<Geometry> {
    if geometry.geometry_type()? == multi {
        let parts = geometry.get_num_geometries()?;
        if parts > 1 {
            return Err(GeoComputeError::Split { parts });
        }
    }
    Ok(geometry)
}

/// Which rows are there, pushed one after the other.
/// A bitmap is only built once a row is null.
struct ValidityBuilder {
    /// How many rows were pushed before the first null one.
    valid: usize,
    rows: usize,
    bitmap: Option<MutableBitmap>,
}

impl ValidityBuilder {
    fn with_capacity(rows: usize) -> Self {
        Self {
            valid: 0,
            rows,
            bitmap: None,
        }
    }

    #[inline(always)]
    fn push(&mut self, is_valid: bool) {
        match &mut self.bitmap {
            Some(bitmap) => bitmap.push(is_valid),
            None if is_valid => self.valid += 1,
            None => {
                let mut bitmap = MutableBitmap::with_capacity(self.rows);
                bitmap.extend_constant(self.valid, true);
                bitmap.push(false);
                self.bitmap = Some(bitmap);
            }
        }
    }

    /// `None` when every row is there.
    fn finish(self) -> Option<Bitmap> {
        self.bitmap.map(Bitmap::from)
    }
}

/// The coordinates the linestring and polygon conversions copy out of GEOS.
/// They drop m, although GEOS can give it (see `into_point_array`).
struct CoordsBuilder {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Empty (and unallocated) without z.
    z: Vec<f64>,
    has_z: bool,
}

impl CoordsBuilder {
    /// With `has_z=true` the coordinates keep their z.
    fn with_capacity(has_z: bool, coords: usize) -> Self {
        Self {
            x: Vec::with_capacity(coords),
            y: Vec::with_capacity(coords),
            z: Vec::with_capacity(if has_z { coords } else { 0 }),
            has_z,
        }
    }

    fn len(&self) -> usize {
        self.x.len()
    }

    /// Appends the coordinates of `geometry`, a linestring or a ring.
    /// `Z` must match the builder.
    fn push<const Z: bool, const REVERSED: bool>(
        &mut self,
        geometry: &impl Geom,
    ) -> GeoComputeResult<()> {
        debug_assert_eq!(Z, self.has_z, "pushed with the wrong z");
        let dim = if Z {
            GeoDimension::XYZ
        } else {
            GeoDimension::XY
        };
        let buffer = geometry.get_coord_seq()?.as_buffer(Some(coord_type(dim)))?;
        let width = dim.field_names().len();
        let push = |coord: &[f64]| {
            self.x.push(coord[0]);
            self.y.push(coord[1]);
            if Z {
                self.z.push(coord[2]);
            }
        };
        if REVERSED {
            buffer.chunks_exact(width).rev().for_each(push);
        } else {
            buffer.chunks_exact(width).for_each(push);
        }
        Ok(())
    }

    fn finish(self) -> ValidCoords {
        ValidCoords::from_axes((self.x, self.y, self.has_z.then_some(self.z), None))
    }
}

/// The offsets of one list level, built by ending one list after the other.
struct OffsetsBuilder {
    offsets: Vec<i64>,
}

impl OffsetsBuilder {
    fn with_capacity(lists: usize) -> Self {
        let mut offsets = Vec::with_capacity(lists + 1);
        offsets.push(0);
        Self { offsets }
    }

    /// How many lists were ended.
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Ends the current list at `end`: the length of the level below, so far.
    #[inline(always)]
    fn push(&mut self, end: usize) {
        debug_assert!(end as i64 >= *self.offsets.last().unwrap());
        self.offsets.push(end as i64);
    }

    fn finish(self) -> OffsetsBuffer<i64> {
        // SAFETY: they start at 0, and every end is the length of a level below that only grows.
        unsafe { Offsets::new_unchecked(self.offsets) }.into()
    }
}
