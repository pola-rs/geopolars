//! Interfacing with the `geos` crate.
//!
//! GEOS, as a C++ library, owns its coordinate storage,
//! so arrow buffers can't be borrowed zero-copy.
//! The copy is one memcpy-like op, so still not very heavy.
//!
//! # Const generic flags
//!
//! ## `EARLY_COORD_CHECK`
//!
//! Converting rows to GEOS, and the convex hulls, are generic over `EARLY_COORD_CHECK`,
//! an [enum@EarlyCoordCheck].
//! A row that fails the check never goes to GEOS.
//! It is an error instead, which `on_row_error` treats like a GEOS error.
//!
//! Converting points takes a bool `EARLY_FINITE_CHECK` instead, which is set unless [EarlyCoordCheck::Off].
//! The two checks are the same for a point,
//! as a point that is itself the empty point passes either check.
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
//! ## `REVERSED_EXTERIOR`, `REVERSED_HOLES`
//!
//! Copying a polygon out of GEOS is generic over `REVERSED_EXTERIOR` and `REVERSED_HOLES`,
//! which `reversed_rings` sets from `new_ring_orientation` and `new_holes`.
//! `CoordsBuilder::push` takes one of them as `REVERSED`, for the ring it copies.
//! A reversed ring is copied in reverse order, which flips its orientation.

mod buffer;
mod convex_hull;
mod interpolate;
mod is_simple;
mod is_topologically_equal;
mod linestrings;
mod offset_curve;
mod overlay;
mod pairs;
mod points;
mod polygons;
mod relate;
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
pub use is_simple::{
    line_strings_are_simple, multi_line_strings_are_simple, multi_polygons_are_simple,
    polygons_are_simple,
};
pub use is_topologically_equal::topologically_equal;
pub use offset_curve::{
    offset_curves_into_line_strings, offset_curves_into_multi_line_strings, OffsetCurveStyle,
};
pub use overlay::{Overlay, OverlayPairs};
pub use relate::{De9im, RelatePairs};
pub use rings_overlap::{rings_overlap_against_one, rings_overlap_pairwise};
pub use shortest_line::{shortest_lines_against_one, shortest_lines_pairwise};
pub use simplify::{
    simplify_line_strings, simplify_multi_line_strings, simplify_multi_polygons, simplify_polygons,
};
pub use topology::{meet_only_at_points_and_no_overlap, polygon_rings_fit};

use std::ops::Range;

use geopolars_views::output::SizeHint;
use geopolars_views::{
    Coords, GeoDimension, GeometryArray, List, MultiPointArray, PointArray, ScalarPoint,
    ValidCoords,
};
use geos::{CoordSeq, CoordType, Geom, Geometry, GeometryTypes};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::{Offsets, OffsetsBuffer};

use crate::{
    row_or_null, EarlyCoordCheck, GeoComputeError, GeoComputeResult, NewHoles, NewRingOrientation,
    OnRowError,
};

/// Converts a single geometry into a GEOS geometry.
/// For an array, see [`ToGeosRows`].
pub trait ToGeos {
    type Geos;

    fn to_geos(&self) -> GeoComputeResult<Self::Geos>;
}

/// Converts an array into one [`GeosRow`] per row.
/// A row without parts is `Empty`, so it never goes to GEOS.
/// A row that fails `early_coord_check`, or that GEOS fails to build, goes to `on_row_error`.
/// The size hint is that of the array itself.
pub trait ToGeosRows {
    fn to_geos_rows(
        &self,
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
    ) -> GeoComputeResult<GeosGeometries> {
        self.to_geos_rows_taking_as_empty(None, on_row_error, early_coord_check)
    }

    /// See [`ToGeosRows::to_geos_rows`].
    /// A row that is set in `as_empty` and not null is also `Empty`, without going to GEOS.
    fn to_geos_rows_taking_as_empty(
        &self,
        as_empty: Option<&Bitmap>,
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
    ) -> GeoComputeResult<GeosGeometries>;
}

impl ToGeosRows for GeometryArray {
    fn to_geos_rows_taking_as_empty(
        &self,
        as_empty: Option<&Bitmap>,
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
    ) -> GeoComputeResult<GeosGeometries> {
        let early = early_coord_check;
        match self {
            GeometryArray::Point(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, on_row_error, early)
            }
            GeometryArray::LineString(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, on_row_error, early)
            }
            GeometryArray::Polygon(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, on_row_error, early)
            }
            GeometryArray::MultiPoint(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, on_row_error, early)
            }
            GeometryArray::MultiLineString(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, on_row_error, early)
            }
            GeometryArray::MultiPolygon(array) => {
                array.to_geos_rows_taking_as_empty(as_empty, on_row_error, early)
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
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
    ) -> GeoComputeResult<GeosGeometries> {
        let geometries = with_const_flags!(
            EARLY_FINITE_CHECK = early_coord_check != EarlyCoordCheck::Off;
            points_to_geos::<EARLY_FINITE_CHECK>(&self.coords, as_empty, on_row_error)
        )?;
        Ok(GeosGeometries {
            geometries,
            size_hint: SizeHint {
                coords: self.coords.len(),
                parts: [0, 0],
            },
            orientation: GeometryOrientation::UNKNOWN,
        })
    }
}

fn points_to_geos<const EARLY_FINITE_CHECK: bool>(
    coords: &Coords,
    as_empty: Option<&Bitmap>,
    on_row_error: OnRowError,
) -> GeoComputeResult<GeosRows> {
    // GEOS has no batch constructor, so only the per-column work is shared.
    let dim = coords.dim();
    let len = dim.field_names().len();
    let coord_type = coord_type(dim);
    let mut geometries = GeosRows::with_capacity(coords.len());
    for i in 0..coords.len() {
        if is_taken_as_empty(as_empty, i) && coords.is_valid(i) {
            geometries.push(Some(GeosRow::Empty));
            continue;
        }
        let point = point_to_geos::<EARLY_FINITE_CHECK>(coords, i, len, coord_type);
        let point = row_or_null(point, on_row_error)?;
        geometries.push(point.map(|point| GeosRow::Geometry {
            geometry: point,
            maybe_wrong_orientation: true,
        }));
    }
    Ok(geometries)
}

impl ToGeosRows for MultiPointArray {
    fn to_geos_rows_taking_as_empty(
        &self,
        as_empty: Option<&Bitmap>,
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
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
            on_row_error,
            early_coord_check,
            |vertices| vertices,
            |vertices| multi_point_to_geos(&axes, vertices),
        )?;
        Ok(GeosGeometries {
            geometries,
            size_hint,
            orientation: GeometryOrientation::UNKNOWN,
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

/// Marks the (valid) rows that don't have any vertices.
/// Apart from points, these are the rows `is_empty` (without `check_nested`) calls empty,
/// such as a polygon whose rings have no vertices.
/// A point has none when its x and y are NaN, as for the conversion to GEOS,
/// even where a z or m that is not NaN makes `is_empty` call it non-empty.
fn rows_without_vertices(array: &GeometryArray) -> Bitmap {
    match array {
        GeometryArray::Point(array) => {
            let coords = &array.coords;
            let is_empty = |i| {
                coords
                    .xy(i)
                    .is_some_and(|(x, y)| is_geoarrow_empty_point(x, y))
            };
            Bitmap::from_iter((0..coords.len()).map(is_empty))
        }
        GeometryArray::LineString(array) => {
            valid_rows_without_vertices(&array.rows, |vertices| vertices)
        }
        GeometryArray::MultiPoint(array) => {
            valid_rows_without_vertices(&array.rows, |points| points)
        }
        GeometryArray::Polygon(array) => {
            valid_rows_without_vertices(&array.rows, |rings| array.rings.span(rings))
        }
        GeometryArray::MultiLineString(array) => {
            valid_rows_without_vertices(&array.rows, |lines| array.lines.span(lines))
        }
        GeometryArray::MultiPolygon(array) => {
            valid_rows_without_vertices(&array.rows, |polygons| {
                array.rings.span(array.polygons.span(polygons))
            })
        }
    }
}

fn valid_rows_without_vertices(
    rows: &List,
    vertices: impl Fn(Range<usize>) -> Range<usize>,
) -> Bitmap {
    // A null row's offsets point anywhere, so its validity is checked first.
    Bitmap::from_iter(
        (0..rows.len()).map(|i| rows.is_valid(i) && vertices(rows.range(i)).is_empty()),
    )
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

/// Converts every row of a list array to GEOS, for [`ToGeosRows::to_geos_rows_taking_as_empty`].
/// `to_geos` builds one geometry from the range of a row's parts.
/// `span` gives where the coordinates of those parts sit in `axes`.
fn rows_to_geos(
    rows: &List,
    axes: &Axes,
    as_empty: Option<&Bitmap>,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
    span: impl Fn(Range<usize>) -> Range<usize>,
    to_geos: impl Fn(Range<usize>) -> GeoComputeResult<Geometry>,
) -> GeoComputeResult<GeosRows> {
    with_const_flags!(
        ALL_VALID = rows.is_all_valid(), EARLY_COORD_CHECK: EarlyCoordCheck = early_coord_check;
        rows_to_geos_with::<ALL_VALID, EARLY_COORD_CHECK>(
            rows,
            axes,
            as_empty,
            on_row_error,
            span,
            to_geos,
        )
    )
}

/// See [`rows_to_geos`].
/// Errors are the exception, so `on_row_error` is checked per row rather than made const.
fn rows_to_geos_with<const ALL_VALID: bool, const EARLY_COORD_CHECK: u8>(
    rows: &List,
    axes: &Axes,
    as_empty: Option<&Bitmap>,
    on_row_error: OnRowError,
    span: impl Fn(Range<usize>) -> Range<usize>,
    to_geos: impl Fn(Range<usize>) -> GeoComputeResult<Geometry>,
) -> GeoComputeResult<GeosRows> {
    let (x, y, _, _) = axes;
    let mut geometries = GeosRows::with_capacity(rows.len());
    for i in 0..rows.len() {
        // A null row's offsets point anywhere, so it is not converted.
        if !ALL_VALID && !rows.is_valid(i) {
            geometries.push(None);
            continue;
        }
        let parts = rows.range(i);
        geometries.push(if parts.is_empty() || is_taken_as_empty(as_empty, i) {
            Some(GeosRow::Empty)
        } else {
            let geometry =
                if const { EarlyCoordCheck::from_u8(EARLY_COORD_CHECK) } == EarlyCoordCheck::Off {
                    to_geos(parts)
                } else {
                    let coords = span(parts.clone());
                    coords_pass_early_check::<EARLY_COORD_CHECK>(&x[coords.clone()], &y[coords])
                        .and_then(|()| to_geos(parts))
                };
            row_or_null(geometry.map(Some), on_row_error)?.map(|geometry| GeosRow::Geometry {
                geometry,
                maybe_wrong_orientation: true,
            })
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
    Err(GeoComputeError::GeosNotFinite { position })
}

#[inline(always)]
fn xy_is_finite_or_empty_or_error(x: &[f64], y: &[f64]) -> GeoComputeResult<()> {
    let passes = |x: f64, y: f64| (x.is_finite() & y.is_finite()) | is_geoarrow_empty_point(x, y);
    // Does not stop at the first that fails, so that it vectorises.
    let every_passes = x.iter().zip(y).fold(true, |ok, (&x, &y)| ok & passes(x, y));
    if every_passes {
        return Ok(());
    }
    let position = x
        .iter()
        .zip(y)
        .map(|(&x, &y)| (x, y))
        .find(|&(x, y)| !passes(x, y))
        .expect("a failing coordinate, as the check above failed");
    Err(GeoComputeError::GeosNotFinite { position })
}

/// Checks the x and y of one row as `EARLY_COORD_CHECK` asks.
#[inline(always)]
fn coords_pass_early_check<const EARLY_COORD_CHECK: u8>(
    x: &[f64],
    y: &[f64],
) -> GeoComputeResult<()> {
    match const { EarlyCoordCheck::from_u8(EARLY_COORD_CHECK) } {
        EarlyCoordCheck::Off => Ok(()),
        EarlyCoordCheck::Finite => xy_is_finite_or_empty_or_error(x, y),
        EarlyCoordCheck::FiniteAndEmpty => xy_is_finite_or_error(x, y),
    }
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
    pub geometries: GeosRows,
    /// Sized for the deepest kind, a multipolygon:
    /// rings (or lines), then polygons.
    pub size_hint: SizeHint<2>,
    /// Holds for every row, except one with `maybe_wrong_orientation=true`.
    pub orientation: GeometryOrientation,
}

/// Which way the rings of a polygon run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Orientation {
    Cw,
    Ccw,
    Unknown,
}

/// Which way the exteriors and the holes of the polygons in a [`GeosGeometries`] run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GeometryOrientation {
    pub outer: Orientation,
    pub holes: Orientation,
}

impl GeometryOrientation {
    pub const UNKNOWN: Self = Self {
        outer: Orientation::Unknown,
        holes: Orientation::Unknown,
    };

    /// The orientation of the rings GEOS builds itself
    /// (see "Orientation of new rings" in `docs/GEOS kernel implementation guide.md`).
    pub const GEOS_BUILT: Self = Self {
        outer: Orientation::Cw,
        holes: Orientation::Ccw,
    };
}

/// One [`GeosRow`] per row, and which rows are null.
pub struct GeosRows {
    /// A null row holds `Empty`, so that a conversion copies nothing for it.
    rows: Vec<GeosRow>,
    validity: ValidityBuilder,
}

impl GeosRows {
    fn with_capacity(rows: usize) -> Self {
        Self {
            rows: Vec::with_capacity(rows),
            validity: ValidityBuilder::with_capacity(rows),
        }
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    #[inline(always)]
    fn push(&mut self, row: Option<GeosRow>) {
        self.validity.push(row.is_some());
        self.rows.push(row.unwrap_or(GeosRow::Empty));
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = Option<&GeosRow>> + Clone {
        let validity = &self.validity;
        let valid = move |(i, row)| validity.is_valid(i).then_some(row);
        self.rows.iter().enumerate().map(valid)
    }
}

impl GeosGeometries {
    /// Replaces every GEOS geometry with what `op` makes of it, given its row index.
    /// This includes a row with `maybe_wrong_orientation`, which then no longer has it.
    /// `op` gives geometries of `orientation`, which the column takes on.
    /// A row `op` fails on goes to `on_row_error`.
    #[inline(always)]
    fn replace_geometries(
        &mut self,
        orientation: GeometryOrientation,
        on_row_error: OnRowError,
        mut op: impl FnMut(usize, &Geometry) -> GeoComputeResult<Geometry>,
    ) -> GeoComputeResult<()> {
        self.orientation = orientation;
        let GeosRows { rows, validity } = &mut self.geometries;
        for (i, row) in rows.iter_mut().enumerate() {
            let GeosRow::Geometry {
                geometry: input, ..
            } = row
            else {
                continue;
            };
            match row_or_null(op(i, input).map(Some), on_row_error)? {
                Some(output) => {
                    *row = GeosRow::Geometry {
                        geometry: output,
                        maybe_wrong_orientation: false,
                    }
                }
                None => {
                    *row = GeosRow::Empty;
                    validity.set_null(i);
                }
            }
        }
        Ok(())
    }
}

/// One row of [`GeosRows`].
pub enum GeosRow {
    /// Empty, without a GEOS geometry. Valid for every kind.
    Empty,
    /// The GEOS representation of a geometry.
    Geometry {
        geometry: Geometry,
        /// The row may not follow [`GeosGeometries::orientation`],
        /// such as a copy of an input row in a column GEOS otherwise built.
        maybe_wrong_orientation: bool,
    },
}

/// Whether to reverse the exterior and the holes of a polygon in a column of `orientation`.
/// A ring of [`Orientation::Unknown`] is never reversed.
fn reversed_rings(
    orientation: GeometryOrientation,
    new_ring_orientation: NewRingOrientation,
    new_holes: NewHoles,
) -> (bool, bool) {
    let (outer, opposite) = match new_ring_orientation {
        NewRingOrientation::Keep => return (false, false),
        NewRingOrientation::Ccw => (Orientation::Ccw, Orientation::Cw),
        NewRingOrientation::Cw => (Orientation::Cw, Orientation::Ccw),
    };
    let holes = match new_holes {
        NewHoles::Opposite => opposite,
        NewHoles::Same => outer,
    };
    let differs = |current, target| current != Orientation::Unknown && current != target;
    (
        differs(orientation.outer, outer),
        differs(orientation.holes, holes),
    )
}

/// Combines the validity of the geometries with that of the distances.
/// It goes on the input before converting it to GEOS,
/// so that a row with a null distance is never checked or converted.
fn validity_with_distances(validity: Option<&Bitmap>, distance: BufferDistance) -> Option<Bitmap> {
    match distance {
        BufferDistance::Constant(_) => validity.cloned(),
        BufferDistance::PerRow(distances) => combine_validities_and(validity, distances.validity()),
    }
}

/// Sends a geometry of a kind the conversion cannot hold to `on_row_error`.
/// `Ok(false)` means its row is null.
fn wrong_kind(
    expected: &str,
    found: GeometryTypes,
    on_row_error: OnRowError,
) -> GeoComputeResult<bool> {
    if on_row_error == OnRowError::Null {
        return Ok(false);
    }
    Err(geos::Error::ConversionError(format!("expected a {expected}, got a {found:?}")).into())
}

/// Refuses a geometry of `multi_type` with several parts, as its single kind cannot hold them.
fn unless_split(geometry: Geometry, multi_type: GeometryTypes) -> GeoComputeResult<Geometry> {
    if geometry.geometry_type()? == multi_type {
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

    /// Row `i` must have been pushed already.
    fn set_null(&mut self, i: usize) {
        let bitmap = self.bitmap.get_or_insert_with(|| {
            let mut bitmap = MutableBitmap::with_capacity(self.rows);
            bitmap.extend_constant(self.valid, true);
            bitmap
        });
        bitmap.set(i, false);
    }

    #[inline(always)]
    fn is_valid(&self, i: usize) -> bool {
        self.bitmap.as_ref().is_none_or(|bitmap| bitmap.get(i))
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
