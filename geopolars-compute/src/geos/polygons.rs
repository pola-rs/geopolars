//! Polygons and multipolygons to GEOS, and back.
//!
//! # Const generic flags
//!
//! See [super] for `Z`, `REVERSED_EXTERIOR` and `REVERSED_HOLES`.

use std::ops::Range;

use geopolars_views::output::SizeHint;
use geopolars_views::{
    List, MultiPolygonArray, PolygonArray, ScalarMultiPolygon, ScalarPolygon, ValidList,
};
use geos::{Geom, Geometry, GeometryTypes};
use polars_arrow::bitmap::Bitmap;

use super::{
    axes, coord_seq, reversed_rings, rows_to_geos, target_orientation, wrong_kind, Axes,
    CoordsBuilder, GeometryOrientation, GeosGeometries, GeosRow, OffsetsBuilder, Orientation,
    ToGeos, ToGeosRows, ValidityBuilder,
};
use crate::{EarlyCoordCheck, GeoComputeResult, NewHoles, NewRingOrientation, OnRowError};

impl ToGeos for ScalarPolygon<'_> {
    type Geos = Geometry;

    fn to_geos(&self) -> GeoComputeResult<Geometry> {
        polygon_to_geos(&axes(self.coords), self.rings, self.parts.clone())
    }
}

impl ToGeosRows for PolygonArray {
    fn to_geos_rows_skipping(
        &self,
        skipped: Option<&Bitmap>,
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
    ) -> GeoComputeResult<GeosGeometries> {
        let axes = axes(&self.coords);
        let size_hint = SizeHint {
            coords: self.coords.len(),
            parts: [self.rings.len(), 0],
        };
        let geometries = rows_to_geos(
            &self.rows,
            &axes,
            skipped,
            on_row_error,
            early_coord_check,
            |parts| self.rings.span(parts),
            |parts| polygon_to_geos(&axes, &self.rings, parts),
        )?;
        Ok(GeosGeometries {
            geometries,
            size_hint,
            orientation: GeometryOrientation::UNKNOWN,
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
    fn to_geos_rows_skipping(
        &self,
        skipped: Option<&Bitmap>,
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
    ) -> GeoComputeResult<GeosGeometries> {
        let axes = axes(&self.coords);
        let size_hint = SizeHint {
            coords: self.coords.len(),
            parts: [self.rings.len(), self.polygons.len()],
        };
        let geometries = rows_to_geos(
            &self.rows,
            &axes,
            skipped,
            on_row_error,
            early_coord_check,
            |parts| self.rings.span(self.polygons.span(parts)),
            |parts| multi_polygon_to_geos(&axes, &self.polygons, &self.rings, parts),
        )?;
        Ok(GeosGeometries {
            geometries,
            size_hint,
            orientation: GeometryOrientation::UNKNOWN,
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
    /// A geometry that is not a polygon goes to `on_row_error`.
    /// The rings are oriented as `new_ring_orientation` and `new_holes` ask,
    /// which relies on `orientation` and `maybe_wrong_orientation` being set
    /// (see "Orientation of new rings" in `docs/source/development/geos-kernel-implementation-guide.md`).
    pub fn into_polygon_array(
        mut self,
        z: bool,
        on_row_error: OnRowError,
        new_ring_orientation: NewRingOrientation,
        new_holes: NewHoles,
    ) -> GeoComputeResult<PolygonArray> {
        let Some(target) = target_orientation(new_ring_orientation, new_holes) else {
            return with_const_flags!(
                Z = z;
                into_polygon_array_with::<Z, false, false, false>(
                    self,
                    GeometryOrientation::UNKNOWN,
                    on_row_error,
                )
            );
        };
        self.flag_every_row_if_unknown();
        let (exterior, holes) = reversed_rings(self.orientation, target);
        with_const_flags!(
            Z = z, REVERSED_EXTERIOR = exterior, REVERSED_HOLES = holes;
            into_polygon_array_with::<Z, REVERSED_EXTERIOR, REVERSED_HOLES, true>(
                self,
                target,
                on_row_error,
            )
        )
    }

    /// See [`GeosGeometries::into_polygon_array`].
    /// GEOS gives a multipolygon of one part as a plain polygon.
    /// So a polygon is read as a single part, or as none if it is empty.
    /// A geometry that is not a polygon or multipolygon fails the same way.
    pub fn into_multi_polygon_array(
        mut self,
        z: bool,
        on_row_error: OnRowError,
        new_ring_orientation: NewRingOrientation,
        new_holes: NewHoles,
    ) -> GeoComputeResult<MultiPolygonArray> {
        let Some(target) = target_orientation(new_ring_orientation, new_holes) else {
            return with_const_flags!(
                Z = z;
                into_multi_polygon_array_with::<Z, false, false, false>(
                    self,
                    GeometryOrientation::UNKNOWN,
                    on_row_error,
                )
            );
        };
        self.flag_every_row_if_unknown();
        let (exterior, holes) = reversed_rings(self.orientation, target);
        with_const_flags!(
            Z = z, REVERSED_EXTERIOR = exterior, REVERSED_HOLES = holes;
            into_multi_polygon_array_with::<Z, REVERSED_EXTERIOR, REVERSED_HOLES, true>(
                self,
                target,
                on_row_error,
            )
        )
    }

    fn flag_every_row_if_unknown(&mut self) {
        let GeometryOrientation { outer, holes } = self.orientation;
        if outer != Orientation::Unknown && holes != Orientation::Unknown {
            return;
        }
        for row in &mut self.geometries {
            if let GeosRow::Geometry {
                maybe_wrong_orientation,
                ..
            } = row
            {
                *maybe_wrong_orientation = true;
            }
        }
    }
}

/// See [`GeosGeometries::into_multi_polygon_array`].
fn into_multi_polygon_array_with<
    const Z: bool,
    const REVERSED_EXTERIOR: bool,
    const REVERSED_HOLES: bool,
    const CHECK_FLAGGED_ORIENTATION: bool,
>(
    geometries: GeosGeometries,
    target: GeometryOrientation,
    on_row_error: OnRowError,
) -> GeoComputeResult<MultiPolygonArray> {
    let GeosGeometries {
        geometries,
        size_hint,
        ..
    } = geometries;
    let mut validity = ValidityBuilder::with_capacity(geometries.len());
    let mut coords = CoordsBuilder::with_capacity(Z, size_hint.coords);
    let mut rings = OffsetsBuilder::with_capacity(size_hint.parts[0]);
    let mut polygons = OffsetsBuilder::with_capacity(size_hint.parts[1]);
    let mut rows = OffsetsBuilder::with_capacity(geometries.len());
    for geometry in geometries {
        let is_valid = match geometry {
            GeosRow::Null | GeosRow::Failed => false,
            GeosRow::Empty => true,
            GeosRow::Skipped => unreachable!("a skipped row never reaches an output"),
            GeosRow::Geometry {
                geometry,
                maybe_wrong_orientation,
            } => {
                if CHECK_FLAGGED_ORIENTATION && maybe_wrong_orientation {
                    push_multi_polygon::<Z, false, false, true>(
                        &mut coords,
                        &mut rings,
                        &mut polygons,
                        &geometry,
                        target,
                        on_row_error,
                    )?
                } else {
                    push_multi_polygon::<Z, REVERSED_EXTERIOR, REVERSED_HOLES, false>(
                        &mut coords,
                        &mut rings,
                        &mut polygons,
                        &geometry,
                        target,
                        on_row_error,
                    )?
                }
            }
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
fn into_polygon_array_with<
    const Z: bool,
    const REVERSED_EXTERIOR: bool,
    const REVERSED_HOLES: bool,
    const CHECK_FLAGGED_ORIENTATION: bool,
>(
    geometries: GeosGeometries,
    target: GeometryOrientation,
    on_row_error: OnRowError,
) -> GeoComputeResult<PolygonArray> {
    let GeosGeometries {
        geometries,
        size_hint,
        ..
    } = geometries;
    let mut validity = ValidityBuilder::with_capacity(geometries.len());
    let mut coords = CoordsBuilder::with_capacity(Z, size_hint.coords);
    let mut rings = OffsetsBuilder::with_capacity(size_hint.parts[0]);
    let mut rows = OffsetsBuilder::with_capacity(geometries.len());
    for geometry in geometries {
        let is_valid = match geometry {
            GeosRow::Null | GeosRow::Failed => false,
            GeosRow::Empty => true,
            GeosRow::Skipped => unreachable!("a skipped row never reaches an output"),
            GeosRow::Geometry {
                geometry: polygon,
                maybe_wrong_orientation,
            } => {
                if CHECK_FLAGGED_ORIENTATION && maybe_wrong_orientation {
                    push_polygon::<Z, false, false, true>(
                        &mut coords,
                        &mut rings,
                        &polygon,
                        target,
                        on_row_error,
                    )?
                } else {
                    push_polygon::<Z, REVERSED_EXTERIOR, REVERSED_HOLES, false>(
                        &mut coords,
                        &mut rings,
                        &polygon,
                        target,
                        on_row_error,
                    )?
                }
            }
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

/// Appends a polygon or a multipolygon as the parts of one multipolygon.
/// Any other kind goes to [wrong_kind], which decides whether its row is valid.
#[inline(always)]
fn push_multi_polygon<
    const Z: bool,
    const REVERSED_EXTERIOR: bool,
    const REVERSED_HOLES: bool,
    const CHECK_ORIENTATION: bool,
>(
    coords: &mut CoordsBuilder,
    rings: &mut OffsetsBuilder,
    polygons: &mut OffsetsBuilder,
    geometry: &Geometry,
    target: GeometryOrientation,
    on_row_error: OnRowError,
) -> GeoComputeResult<bool> {
    match geometry.geometry_type()? {
        GeometryTypes::Polygon if geometry.is_empty()? => Ok(true),
        GeometryTypes::Polygon => {
            push_rings::<Z, REVERSED_EXTERIOR, REVERSED_HOLES, CHECK_ORIENTATION>(
                coords, rings, geometry, target,
            )?;
            polygons.push(rings.len());
            Ok(true)
        }
        GeometryTypes::MultiPolygon => {
            for n in 0..geometry.get_num_geometries()? {
                let polygon = geometry.get_geometry_n(n)?;
                push_rings::<Z, REVERSED_EXTERIOR, REVERSED_HOLES, CHECK_ORIENTATION>(
                    coords, rings, &polygon, target,
                )?;
                polygons.push(rings.len());
            }
            Ok(true)
        }
        other => wrong_kind("multipolygon", other, on_row_error),
    }
}

/// Appends a polygon.
/// Any other kind goes to [wrong_kind], which decides whether its row is valid.
#[inline(always)]
fn push_polygon<
    const Z: bool,
    const REVERSED_EXTERIOR: bool,
    const REVERSED_HOLES: bool,
    const CHECK_ORIENTATION: bool,
>(
    coords: &mut CoordsBuilder,
    rings: &mut OffsetsBuilder,
    geometry: &Geometry,
    target: GeometryOrientation,
    on_row_error: OnRowError,
) -> GeoComputeResult<bool> {
    match geometry.geometry_type()? {
        GeometryTypes::Polygon => {
            push_rings::<Z, REVERSED_EXTERIOR, REVERSED_HOLES, CHECK_ORIENTATION>(
                coords, rings, geometry, target,
            )?;
            Ok(true)
        }
        other => wrong_kind("polygon", other, on_row_error),
    }
}

/// Appends the rings of `polygon`, each ended in `rings`.
/// The exterior comes first, then the interiors.
/// An empty polygon has no rings.
/// With `CHECKED=true`, each ring is checked against `target` instead.
#[inline(always)]
fn push_rings<
    const Z: bool,
    const REVERSED_EXTERIOR: bool,
    const REVERSED_HOLES: bool,
    const CHECK_ORIENTATION: bool,
>(
    coords: &mut CoordsBuilder,
    rings: &mut OffsetsBuilder,
    polygon: &impl Geom,
    target: GeometryOrientation,
) -> GeoComputeResult<()> {
    // GEOS gives an empty polygon an empty exterior.
    // We would store that as a ring without vertices.
    if polygon.is_empty()? {
        return Ok(());
    }
    let exterior = polygon.get_exterior_ring()?;
    push_ring::<Z, REVERSED_EXTERIOR, CHECK_ORIENTATION>(coords, &exterior, target.outer)?;
    rings.push(coords.len());
    for n in 0..polygon.get_num_interior_rings()? {
        let hole = polygon.get_interior_ring_n(n)?;
        push_ring::<Z, REVERSED_HOLES, CHECK_ORIENTATION>(coords, &hole, target.holes)?;
        rings.push(coords.len());
    }
    Ok(())
}

#[inline(always)]
fn push_ring<const Z: bool, const REVERSED: bool, const CHECK_ORIENTATION: bool>(
    coords: &mut CoordsBuilder,
    ring: &impl Geom,
    target: Orientation,
) -> GeoComputeResult<()> {
    if CHECK_ORIENTATION {
        coords.push_running::<Z>(ring, target)
    } else {
        coords.push::<Z, REVERSED>(ring)
    }
}
