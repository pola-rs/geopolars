//! The intersection of two geometries, forwarded to GEOS.
//! See `intersection` in Python for the rules.
//!
//! GEOS gives a result of any kind per row, mixed ones as a `GeometryCollection`.
//! Each result is narrowed down to the parts of one dimension ([`IntersectionParts`]),
//! so that every row fits the one output kind.

use geopolars_core::output::SizeHint;
use geopolars_core::{GeoDimension, GeometryArray};
use geos::{Geom, Geometry, GeometryTypes};
use polars_arrow::array::Array;

use super::{GeosGeometries, GeosRow, ToGeosRows};
use crate::{row_or_null, GeoComputeError, GeoComputeResult};

/// The dimension of the parts an intersection keeps.
/// Ordered by dimension, from points up to polygons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntersectionParts {
    Points,
    Lines,
    Polygons,
}

impl IntersectionParts {
    /// The GEOS types of a single part, and of several.
    fn geometry_types(self) -> (GeometryTypes, GeometryTypes) {
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

/// What an intersection gives: which parts, in which kind.
#[derive(Debug, Clone, Copy)]
pub struct IntersectionOutput {
    pub parts: IntersectionParts,
    /// The multi kind of `parts`.
    /// Otherwise the single kind, and a result in several parts is an error.
    pub multi: bool,
    /// Keeps z; m is always dropped.
    pub z: bool,
    /// Reverses every ring of a polygon, so its exterior runs counter-clockwise.
    /// Otherwise it runs clockwise, as GEOS gives it.
    pub ccw_outer: bool,
}

/// The intersection of every row of `a` with the same row of `b`, forwarded to GEOS,
/// as the array `output` says (a point array, or a list array for any other kind).
/// A null row on either side gives a null row, and an empty one an empty row.
/// If `ignore_per_row_errors=true`, a row that fails is null;
/// otherwise its error is returned.
/// A row fails when GEOS fails on it,
/// or when it splits into several parts for a single kind ([`GeoComputeError::Split`]).
/// Each side is converted to GEOS in full, so a row that is null on only one side can still fail.
///
/// `a` and `b` must have the same length.
pub fn intersection_pairwise(
    a: &GeometryArray,
    b: &GeometryArray,
    output: IntersectionOutput,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<Box<dyn Array>> {
    let a = a.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let b = b.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    debug_assert_eq!(
        a.geometries.len(),
        b.geometries.len(),
        "`a` and `b` must have the same length"
    );
    let intersections = a
        .geometries
        .iter()
        .zip(&b.geometries)
        .map(|(a, b)| intersect_rows(a, b, output, ignore_per_row_errors))
        .collect::<GeoComputeResult<Vec<_>>>()?;
    into_array(
        intersections,
        sum(a.size_hint, b.size_hint),
        output,
        ignore_per_row_errors,
    )
}

/// Like [intersection_pairwise], with a single geometry against every row of `many`.
/// If `one_is_a=true`, `one` is the first geometry of each intersection, otherwise the second.
/// As in [intersection_pairwise], `many` is converted to GEOS in full, even if `one` is null.
///
/// `one` must hold a single row.
pub fn intersection_against_one(
    one: &GeometryArray,
    many: &GeometryArray,
    one_is_a: bool,
    output: IntersectionOutput,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<Box<dyn Array>> {
    let one = one.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let [one_row] = one.geometries.as_slice() else {
        panic!("`one` must be a single row, got {}", one.geometries.len());
    };
    let many = many.to_geos_rows(ignore_per_row_errors, early_finite_check)?;
    let intersections = many
        .geometries
        .iter()
        .map(|row| {
            let (a, b) = if one_is_a {
                (one_row, row)
            } else {
                (row, one_row)
            };
            intersect_rows(a, b, output, ignore_per_row_errors)
        })
        .collect::<GeoComputeResult<Vec<_>>>()?;
    into_array(
        intersections,
        sum(one.size_hint, many.size_hint),
        output,
        ignore_per_row_errors,
    )
}

/// Overlay can add vertices where the two cross,
/// so this is only a starting capacity, not an upper bound.
fn sum(a: SizeHint<2>, b: SizeHint<2>) -> SizeHint<2> {
    SizeHint {
        coords: a.coords + b.coords,
        parts: [a.parts[0] + b.parts[0], a.parts[1] + b.parts[1]],
    }
}

#[inline(always)]
fn intersect_rows(
    a: &GeosRow,
    b: &GeosRow,
    output: IntersectionOutput,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<GeosRow> {
    let intersection = match (a, b) {
        (GeosRow::Null, _) | (_, GeosRow::Null) => return Ok(GeosRow::Null),
        // An empty row never went to GEOS, and has nothing in common with anything.
        (GeosRow::Empty, _) | (_, GeosRow::Empty) => return Ok(GeosRow::Empty),
        (GeosRow::Geometry(a), GeosRow::Geometry(b)) => a
            .intersection(b)
            .map_err(GeoComputeError::from)
            .and_then(|intersection| kept_parts(intersection, output)),
    };
    Ok(row_or_null(intersection.map(Some), ignore_per_row_errors)?.unwrap_or(GeosRow::Null))
}

/// The parts of `intersection` of the dimension `output` keeps, as one geometry of that kind.
/// For the single kind, a single part is passed on as is,
/// and several parts are a [`GeoComputeError::Split`].
/// Parts of another dimension are dropped.
fn kept_parts(intersection: Geometry, output: IntersectionOutput) -> GeoComputeResult<GeosRow> {
    if intersection.is_empty()? {
        return Ok(GeosRow::Empty);
    }
    let (single, multi) = output.parts.geometry_types();
    let found = intersection.geometry_type()?;
    // The common cases: GEOS gave the kept dimension only, so nothing is copied.
    if found == single || (found == multi && output.multi) {
        return Ok(GeosRow::Geometry(intersection));
    }
    if found != multi && found != GeometryTypes::GeometryCollection {
        return Ok(GeosRow::Empty);
    }
    // A collection mixes dimensions, and a multi kind can have a single part to take out.
    // Only these copy their kept parts.
    let mut parts = Vec::new();
    for n in 0..intersection.get_num_geometries()? {
        let part = intersection.get_geometry_n(n)?;
        if part.geometry_type()? == single && !part.is_empty()? {
            parts.push(Geom::clone(&part)?);
        }
    }
    match parts.len() {
        0 => Ok(GeosRow::Empty),
        _ if output.multi => Ok(GeosRow::Geometry(output.parts.create_multi(parts)?)),
        1 => Ok(GeosRow::Geometry(parts.pop().expect("one part"))),
        parts => Err(GeoComputeError::Split { parts }),
    }
}

fn into_array(
    geometries: Vec<GeosRow>,
    size_hint: SizeHint<2>,
    output: IntersectionOutput,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<Box<dyn Array>> {
    let geometries = GeosGeometries {
        geometries,
        size_hint,
    };
    let IntersectionOutput {
        parts,
        multi,
        z,
        ccw_outer,
    } = output;
    let ignore = ignore_per_row_errors;
    // `kept_parts` only lets through what these take, so none of them meets a wrong kind.
    Ok(match (parts, multi) {
        (IntersectionParts::Points, false) => {
            let dim = if z {
                GeoDimension::XYZ
            } else {
                GeoDimension::XY
            };
            Box::new(geometries.into_point_array(dim)?.into_struct_array())
        }
        (IntersectionParts::Points, true) => with_const_flags!(
            Z = z;
            Box::new(geometries.into_multi_point_array::<Z>(ignore)?.into_list_array())
        ),
        (IntersectionParts::Lines, false) => with_const_flags!(
            Z = z;
            Box::new(geometries.into_line_string_array::<Z>(ignore)?.into_list_array())
        ),
        (IntersectionParts::Lines, true) => with_const_flags!(
            Z = z;
            Box::new(geometries.into_multi_line_string_array::<Z>(ignore)?.into_list_array())
        ),
        (IntersectionParts::Polygons, false) => Box::new(
            geometries
                .into_polygon_array(z, ignore, ccw_outer)?
                .into_list_array(),
        ),
        (IntersectionParts::Polygons, true) => Box::new(
            geometries
                .into_multi_polygon_array(z, ignore, ccw_outer)?
                .into_list_array(),
        ),
    })
}
