//! Polygons and multipolygons to GEOS, and back.
//!
//! # Const generic flags
//!
//! See [super] for `Z` and `REVERSED`.

use std::ops::Range;

use geopolars_core::output::SizeHint;
use geopolars_core::{
    List, MultiPolygonArray, PolygonArray, ScalarMultiPolygon, ScalarPolygon, ValidList,
};
use geos::{Geom, Geometry, GeometryTypes};
use polars_arrow::bitmap::Bitmap;

use super::{
    axes, coord_seq, rows_to_geos, wrong_kind, Axes, CoordsBuilder, GeosGeometries, GeosRow,
    OffsetsBuilder, ToGeos, ToGeosRows, ValidityBuilder,
};
use crate::GeoComputeResult;

impl ToGeos for ScalarPolygon<'_> {
    type Geos = Geometry;

    fn to_geos(&self) -> GeoComputeResult<Geometry> {
        polygon_to_geos(&axes(self.coords), self.rings, self.parts.clone())
    }
}

impl ToGeosRows for PolygonArray {
    fn to_geos_rows_taking_as_empty(
        &self,
        as_empty: Option<&Bitmap>,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        let axes = axes(&self.coords);
        let size_hint = SizeHint {
            coords: self.coords.len(),
            parts: [self.rings.len(), 0],
        };
        let geometries = rows_to_geos(
            &self.rows,
            &axes,
            as_empty,
            ignore_per_row_errors,
            early_finite_check,
            |parts| self.rings.span(parts),
            |parts| polygon_to_geos(&axes, &self.rings, parts),
        )?;
        Ok(GeosGeometries {
            geometries,
            size_hint,
        })
    }
}

impl ToGeos for ScalarMultiPolygon<'_> {
    type Geos = Geometry;

    fn to_geos(&self) -> GeoComputeResult<Geometry> {
        multi_polygon_to_geos(
            &axes(self.coords),
            self.polygons,
            self.rings,
            self.parts.clone(),
        )
    }
}

impl ToGeosRows for MultiPolygonArray {
    fn to_geos_rows_taking_as_empty(
        &self,
        as_empty: Option<&Bitmap>,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        let axes = axes(&self.coords);
        let size_hint = SizeHint {
            coords: self.coords.len(),
            parts: [self.rings.len(), self.polygons.len()],
        };
        let geometries = rows_to_geos(
            &self.rows,
            &axes,
            as_empty,
            ignore_per_row_errors,
            early_finite_check,
            |parts| self.rings.span(self.polygons.span(parts)),
            |parts| multi_polygon_to_geos(&axes, &self.polygons, &self.rings, parts),
        )?;
        Ok(GeosGeometries {
            geometries,
            size_hint,
        })
    }
}

/// The polygon of `rings` `parts`, the first of which is the exterior.
/// No parts gives an empty polygon.
#[inline(always)]
pub(super) fn polygon_to_geos(
    axes: &Axes,
    rings: &ValidList,
    parts: Range<usize>,
) -> GeoComputeResult<Geometry> {
    let mut parts = parts.map(|j| linear_ring_to_geos(axes, rings.range(j)));
    // GEOS can't build a polygon without an exterior, so an empty one has its own constructor.
    let Some(exterior) = parts.next() else {
        return Ok(Geometry::create_empty_polygon()?);
    };
    Ok(Geometry::create_polygon(
        exterior?,
        parts.collect::<GeoComputeResult<_>>()?,
    )?)
}

#[inline(always)]
pub(super) fn polygon_enclosed_by_ring_to_geos(
    axes: &Axes,
    vertices: Range<usize>,
) -> GeoComputeResult<Geometry> {
    Ok(Geometry::create_polygon(
        linear_ring_to_geos(axes, vertices)?,
        vec![],
    )?)
}

#[inline(always)]
fn linear_ring_to_geos(axes: &Axes, vertices: Range<usize>) -> GeoComputeResult<Geometry> {
    Ok(Geometry::create_linear_ring(coord_seq(axes, vertices)?)?)
}

/// The multipolygon of `polygons` `parts`.
/// No parts gives an empty multipolygon.
#[inline(always)]
pub(super) fn multi_polygon_to_geos(
    axes: &Axes,
    polygons: &ValidList,
    rings: &ValidList,
    parts: Range<usize>,
) -> GeoComputeResult<Geometry> {
    // GEOS takes ownership of the parts, so they can't share a buffer across rows.
    let mut geometries = Vec::with_capacity(parts.len());
    for j in parts {
        geometries.push(polygon_to_geos(axes, rings, polygons.range(j))?);
    }
    Ok(Geometry::create_multipolygon(geometries)?)
}

impl GeosGeometries {
    /// Our polygon array of the geometries, one row each.
    /// Every GEOS geometry is freed once it is copied.
    /// A geometry that is not a polygon is an error.
    /// If `ignore_per_row_errors=true`, its row is null instead.
    pub fn into_polygon_array(
        self,
        z: bool,
        ignore_per_row_errors: bool,
        reversed: bool,
    ) -> GeoComputeResult<PolygonArray> {
        with_const_flags!(
            Z = z, REVERSED = reversed;
            into_polygon_array_with::<Z, REVERSED>(self, ignore_per_row_errors)
        )
    }

    /// See [`GeosGeometries::into_polygon_array`].
    /// GEOS gives a multipolygon of one part as a plain polygon.
    /// So a polygon is read as a single part, or as none if it is empty.
    /// A geometry that is not a polygon or multipolygon fails the same way.
    pub fn into_multi_polygon_array(
        self,
        z: bool,
        ignore_per_row_errors: bool,
        reversed: bool,
    ) -> GeoComputeResult<MultiPolygonArray> {
        with_const_flags!(
            Z = z, REVERSED = reversed;
            into_multi_polygon_array_with::<Z, REVERSED>(self, ignore_per_row_errors)
        )
    }
}

/// See [`GeosGeometries::into_multi_polygon_array`].
fn into_multi_polygon_array_with<const Z: bool, const REVERSED: bool>(
    geometries: GeosGeometries,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<MultiPolygonArray> {
    let mut validity = ValidityBuilder::with_capacity(geometries.geometries.len());
    let mut coords = CoordsBuilder::with_capacity(Z, geometries.size_hint.coords);
    let mut rings = OffsetsBuilder::with_capacity(geometries.size_hint.parts[0]);
    let mut polygons = OffsetsBuilder::with_capacity(geometries.size_hint.parts[1]);
    let mut rows = OffsetsBuilder::with_capacity(geometries.geometries.len());
    for geometry in geometries.geometries {
        let is_valid = match geometry {
            GeosRow::Null => false,
            GeosRow::Empty => true,
            GeosRow::Geometry(geometry) => match geometry.geometry_type()? {
                GeometryTypes::Polygon if geometry.is_empty()? => true,
                GeometryTypes::Polygon => {
                    push_rings::<Z, REVERSED>(&mut coords, &mut rings, &geometry)?;
                    polygons.push(rings.len());
                    true
                }
                GeometryTypes::MultiPolygon => {
                    for n in 0..geometry.get_num_geometries()? {
                        push_rings::<Z, REVERSED>(
                            &mut coords,
                            &mut rings,
                            &geometry.get_geometry_n(n)?,
                        )?;
                        polygons.push(rings.len());
                    }
                    true
                }
                other => wrong_kind("multipolygon", other, ignore_per_row_errors)?,
            },
        };
        validity.push(is_valid);
        rows.push(polygons.len());
    }
    Ok(MultiPolygonArray::from_parts(
        List::from_offsets(rows.finish(), validity.finish()),
        ValidList::from_offsets(polygons.finish()),
        ValidList::from_offsets(rings.finish()),
        coords.finish(),
    ))
}

/// See [`GeosGeometries::into_polygon_array`].
fn into_polygon_array_with<const Z: bool, const REVERSED: bool>(
    geometries: GeosGeometries,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<PolygonArray> {
    let mut validity = ValidityBuilder::with_capacity(geometries.geometries.len());
    let mut coords = CoordsBuilder::with_capacity(Z, geometries.size_hint.coords);
    let mut rings = OffsetsBuilder::with_capacity(geometries.size_hint.parts[0]);
    let mut rows = OffsetsBuilder::with_capacity(geometries.geometries.len());
    for geometry in geometries.geometries {
        let is_valid = match geometry {
            GeosRow::Null => false,
            GeosRow::Empty => true,
            GeosRow::Geometry(polygon) => match polygon.geometry_type()? {
                GeometryTypes::Polygon => {
                    push_rings::<Z, REVERSED>(&mut coords, &mut rings, &polygon)?;
                    true
                }
                other => wrong_kind("polygon", other, ignore_per_row_errors)?,
            },
        };
        validity.push(is_valid);
        rows.push(rings.len());
    }
    Ok(PolygonArray::from_parts(
        List::from_offsets(rows.finish(), validity.finish()),
        ValidList::from_offsets(rings.finish()),
        coords.finish(),
    ))
}

/// Appends the rings of `polygon`, each ended in `rings`.
/// The exterior comes first, then the interiors.
/// An empty polygon has no rings.
#[inline(always)]
fn push_rings<const Z: bool, const REVERSED: bool>(
    coords: &mut CoordsBuilder,
    rings: &mut OffsetsBuilder,
    polygon: &impl Geom,
) -> GeoComputeResult<()> {
    // GEOS gives an empty polygon an empty exterior.
    // We would store that as a ring without vertices.
    if polygon.is_empty()? {
        return Ok(());
    }
    coords.push::<Z, REVERSED>(&polygon.get_exterior_ring()?)?;
    rings.push(coords.len());
    for n in 0..polygon.get_num_interior_rings()? {
        coords.push::<Z, REVERSED>(&polygon.get_interior_ring_n(n)?)?;
        rings.push(coords.len());
    }
    Ok(())
}
