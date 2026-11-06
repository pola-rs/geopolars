//! The intersection of two geometries, forwarded to GEOS.
//! See `intersection` in Python for the rules.
//!
//! GEOS gives a result of any kind per row, mixed ones as a `GeometryCollection`.
//! The caller picks the output kind, through one of the `into_*` methods of [`IntersectionPairs`].
//! Each of those narrows every result down to the parts of its dimension,
//! so that every row fits that kind.

use geopolars_core::output::SizeHint;
use geopolars_core::{
    GeoDimension, GeometryArray, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use geos::{Geom, Geometry, GeometryTypes};

use super::{GeosGeometries, GeosRow, ToGeosRows};
use crate::{row_or_null, GeoComputeError, GeoComputeResult};

/// The dimension of the parts an intersection keeps.
#[derive(Debug, Clone, Copy)]
enum IntersectionParts {
    Points,
    Lines,
    Polygons,
}

impl IntersectionParts {
    fn single_and_multi_types(self) -> (GeometryTypes, GeometryTypes) {
        match self {
            Self::Points => (GeometryTypes::Point, GeometryTypes::MultiPoint),
            Self::Lines => (GeometryTypes::LineString, GeometryTypes::MultiLineString),
            Self::Polygons => (GeometryTypes::Polygon, GeometryTypes::MultiPolygon),
        }
    }

    fn create_multi(self, parts: Vec<Geometry>) -> GeoComputeResult<Geometry> {
        Ok(match self {
            Self::Points => Geometry::create_multipoint(parts)?,
            Self::Lines => Geometry::create_multiline_string(parts)?,
            Self::Polygons => Geometry::create_multipolygon(parts)?,
        })
    }
}

/// Two arrays of geometries converted to GEOS, with their rows paired up to intersect.
/// A single row on either side is paired with every row of the other.
/// Each `into_*` method intersects them, narrows every intersection down to its own kind,
/// and copies them into our array of it.
/// m is always dropped, whatever the inputs have.
pub struct IntersectionPairs {
    a: Vec<GeosRow>,
    b: Vec<GeosRow>,
    rows: usize,
    size_hint: SizeHint<2>,
    ignore_per_row_errors: bool,
}

// TODO: `IntersectionPairs` pairs its GEOS rows
// the same way as `topologically_equal_pairwise` and `topologically_equal_against_one`.
// Turn that into a shared helper once a third GEOS operation on two geometries
// (such as `union` or `difference`) needs it.

impl IntersectionPairs {
    /// Implements the Python `intersection`, which documents its behaviour.
    /// Each side is converted to GEOS in full, so a row paired with a null can still fail.
    /// If `ignore_per_row_errors=true`, a row that fails here or in the `into_*` method is null.
    ///
    /// `a` and `b` must have the same length, or one of them a single row.
    pub fn new(
        a: &GeometryArray,
        b: &GeometryArray,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<Self> {
        let a = a.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
        let b = b.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
        let (a_rows, b_rows) = (a.geometries.len(), b.geometries.len());
        let rows = if a_rows == 1 { b_rows } else { a_rows };
        assert!(
            b_rows == rows || b_rows == 1,
            "cannot pair {a_rows} geometries with {b_rows}"
        );
        Ok(Self {
            a: a.geometries,
            b: b.geometries,
            rows,
            size_hint: summed_size_hint(a.size_hint, b.size_hint),
            ignore_per_row_errors,
        })
    }
    /// As points, the empty point where there are none.
    /// Several points fail with [`GeoComputeError::Split`].
    pub fn into_points(self, z: bool) -> GeoComputeResult<PointArray> {
        let dim = if z {
            GeoDimension::XYZ
        } else {
            GeoDimension::XY
        };
        self.intersected(IntersectionParts::Points, false)?
            .into_point_array(dim)
    }

    pub fn into_multi_points(self, z: bool) -> GeoComputeResult<MultiPointArray> {
        let intersections = self.intersected(IntersectionParts::Points, true)?;
        with_const_flags!(Z = z; intersections.into_multi_point_array::<Z>(self.ignore_per_row_errors))
    }

    /// Several linestrings fail with [`GeoComputeError::Split`].
    pub fn into_line_strings(self, z: bool) -> GeoComputeResult<LineStringArray> {
        let intersections = self.intersected(IntersectionParts::Lines, false)?;
        with_const_flags!(Z = z; intersections.into_line_string_array::<Z>(self.ignore_per_row_errors))
    }

    pub fn into_multi_line_strings(self, z: bool) -> GeoComputeResult<MultiLineStringArray> {
        let intersections = self.intersected(IntersectionParts::Lines, true)?;
        with_const_flags!(Z = z; intersections.into_multi_line_string_array::<Z>(self.ignore_per_row_errors))
    }

    /// Several polygons fail with [`GeoComputeError::Split`].
    /// GEOS gives a clockwise exterior, so `ccw_outer=true` reverses every ring.
    pub fn into_polygons(self, z: bool, ccw_outer: bool) -> GeoComputeResult<PolygonArray> {
        self.intersected(IntersectionParts::Polygons, false)?
            .into_polygon_array(z, self.ignore_per_row_errors, ccw_outer)
    }

    /// See [`IntersectionPairs::into_polygons`] for `ccw_outer`.
    pub fn into_multi_polygons(
        self,
        z: bool,
        ccw_outer: bool,
    ) -> GeoComputeResult<MultiPolygonArray> {
        self.intersected(IntersectionParts::Polygons, true)?
            .into_multi_polygon_array(z, self.ignore_per_row_errors, ccw_outer)
    }

    /// Every pair intersected, and narrowed down to its `parts`,
    /// which the `into_*_array` of that kind takes:
    /// a single geometry, or with `multi=true` a single or a multi one.
    fn intersected(
        &self,
        parts: IntersectionParts,
        multi: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        let ignore = self.ignore_per_row_errors;
        let geometries = broadcast(&self.a, self.rows)
            .zip(broadcast(&self.b, self.rows))
            .map(|(a, b)| intersect_rows(a, b, parts, multi, ignore))
            .collect::<GeoComputeResult<Vec<_>>>()?;
        Ok(GeosGeometries {
            geometries,
            size_hint: self.size_hint,
        })
    }
}

/// `len` rows out of `rows`, which holds that many or a single one to repeat.
fn broadcast(rows: &[GeosRow], len: usize) -> impl Iterator<Item = &GeosRow> {
    rows.iter().cycle().take(len)
}

fn summed_size_hint(a: SizeHint<2>, b: SizeHint<2>) -> SizeHint<2> {
    // Overlay adds a vertex wherever the two cross,
    // so the result can have more vertices than both inputs together.
    SizeHint {
        coords: a.coords + b.coords,
        parts: [a.parts[0] + b.parts[0], a.parts[1] + b.parts[1]],
    }
}

#[inline(always)]
fn intersect_rows(
    a: &GeosRow,
    b: &GeosRow,
    parts: IntersectionParts,
    multi: bool,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<GeosRow> {
    let intersection = match (a, b) {
        (GeosRow::Null, _) | (_, GeosRow::Null) => return Ok(GeosRow::Null),
        // An empty row never went to GEOS, and has nothing in common with anything.
        (GeosRow::Empty, _) | (_, GeosRow::Empty) => return Ok(GeosRow::Empty),
        (GeosRow::Geometry(a), GeosRow::Geometry(b)) => a
            .intersection(b)
            .map_err(GeoComputeError::from)
            .and_then(|intersection| kept_parts(intersection, parts, multi)),
    };
    Ok(row_or_null(intersection.map(Some), ignore_per_row_errors)?.unwrap_or(GeosRow::Null))
}

/// Narrows an intersection down to its `parts`,
/// as one geometry if `multi=false` (or [`GeoComputeError::Split`] if it has several).
/// With `multi=true`, a single part can stay a single geometry.
fn kept_parts(
    intersection: Geometry,
    parts: IntersectionParts,
    multi: bool,
) -> GeoComputeResult<GeosRow> {
    if intersection.is_empty()? {
        return Ok(GeosRow::Empty);
    }
    let (single_type, multi_type) = parts.single_and_multi_types();
    let found = intersection.geometry_type()?;
    // The common cases: GEOS gave the kept dimension only, so nothing is copied.
    if found == single_type || (found == multi_type && multi) {
        return Ok(GeosRow::Geometry(intersection));
    }
    if found != multi_type && found != GeometryTypes::GeometryCollection {
        return Ok(GeosRow::Empty);
    }
    // A collection can mix dimensions, and a multi geometry of a single part has to be unpacked.
    // Only these copy their kept parts.
    let mut kept = Vec::new();
    for n in 0..intersection.get_num_geometries()? {
        let part = intersection.get_geometry_n(n)?;
        if part.geometry_type()? == single_type && !part.is_empty()? {
            kept.push(Geom::clone(&part)?);
        }
    }
    match kept.len() {
        0 => Ok(GeosRow::Empty),
        _ if multi => Ok(GeosRow::Geometry(parts.create_multi(kept)?)),
        1 => Ok(GeosRow::Geometry(kept.pop().expect("one part"))),
        parts => Err(GeoComputeError::Split { parts }),
    }
}
