//! Linestrings and multilinestrings to GEOS, and back.
//!
//! # Const generic flags
//!
//! See [super] for `Z`.

use std::ops::Range;

use geopolars_core::output::SizeHint;
use geopolars_core::{
    LineStringArray, List, MultiLineStringArray, ScalarLineString, ScalarMultiLineString, ValidList,
};
use geos::{Geom, Geometry, GeometryTypes};

use super::{
    axes, coord_seq, rows_to_geos, wrong_kind, Axes, CoordsBuilder, GeosGeometries, GeosRow,
    OffsetsBuilder, ToGeos, ToGeosRows, ValidityBuilder,
};
use crate::GeoComputeResult;

impl ToGeos for ScalarLineString<'_> {
    type Geos = Geometry;

    fn to_geos(&self) -> GeoComputeResult<Geometry> {
        line_string_to_geos(&axes(self.coords), self.vertices.clone())
    }
}

impl ToGeosRows for LineStringArray {
    fn to_geos_rows(
        &self,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        let axes = axes(&self.coords);
        let size_hint = SizeHint {
            coords: self.coords.len(),
            parts: [0, 0],
        };
        rows_to_geos(
            &self.rows,
            &axes,
            size_hint,
            ignore_per_row_errors,
            early_finite_check,
            |vertices| vertices,
            |vertices| line_string_to_geos(&axes, vertices),
        )
    }
}

impl ToGeos for ScalarMultiLineString<'_> {
    type Geos = Geometry;

    fn to_geos(&self) -> GeoComputeResult<Geometry> {
        multi_line_string_to_geos(&axes(self.coords), self.lines, self.parts.clone(), false)
    }
}

impl ToGeosRows for MultiLineStringArray {
    fn to_geos_rows(
        &self,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        multi_line_strings_to_geos_rows(self, false, ignore_per_row_errors, early_finite_check)
    }
}

/// Like [`ToGeosRows::to_geos_rows`].
/// If `drop_empty_parts=true`, a part without vertices is left out.
pub(super) fn multi_line_strings_to_geos_rows(
    array: &MultiLineStringArray,
    drop_empty_parts: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<GeosGeometries> {
    let axes = axes(&array.coords);
    let size_hint = SizeHint {
        coords: array.coords.len(),
        parts: [array.lines.len(), 0],
    };
    rows_to_geos(
        &array.rows,
        &axes,
        size_hint,
        ignore_per_row_errors,
        early_finite_check,
        |parts| array.lines.span(parts),
        |parts| multi_line_string_to_geos(&axes, &array.lines, parts, drop_empty_parts),
    )
}

/// Can copy from axes in one go.
/// No vertices gives an empty line string.
#[inline(always)]
pub(super) fn line_string_to_geos(
    axes: &Axes,
    vertices: Range<usize>,
) -> GeoComputeResult<Geometry> {
    Ok(Geometry::create_line_string(coord_seq(axes, vertices)?)?)
}

/// No parts gives an empty multilinestring.
/// If `drop_empty_parts=true`, a part without vertices is left out.
#[inline(always)]
pub(super) fn multi_line_string_to_geos(
    axes: &Axes,
    lines: &ValidList,
    parts: Range<usize>,
    drop_empty_parts: bool,
) -> GeoComputeResult<Geometry> {
    // GEOS takes ownership of the parts, so they can't share a buffer across rows.
    let mut line_strings = Vec::with_capacity(parts.len());
    for j in parts {
        let vertices = lines.range(j);
        if drop_empty_parts && vertices.is_empty() {
            continue;
        }
        line_strings.push(line_string_to_geos(axes, vertices)?);
    }
    Ok(Geometry::create_multiline_string(line_strings)?)
}

impl GeosGeometries {
    /// Our linestring array of the geometries, one row each.
    /// Every GEOS geometry is freed once it is copied.
    /// A geometry that is not a linestring is an error.
    /// If `ignore_per_row_errors=true`, its row is null instead.
    pub fn into_line_string_array<const Z: bool>(
        self,
        ignore_per_row_errors: bool,
    ) -> GeoComputeResult<LineStringArray> {
        let mut validity = ValidityBuilder::with_capacity(self.geometries.len());
        let mut coords = CoordsBuilder::with_capacity(Z, self.size_hint.coords);
        let mut rows = OffsetsBuilder::with_capacity(self.geometries.len());
        for geometry in self.geometries {
            let is_valid = match geometry {
                GeosRow::Null => false,
                GeosRow::Empty => true,
                GeosRow::Geometry(line_string) => match line_string.geometry_type()? {
                    GeometryTypes::LineString => {
                        coords.push::<Z, false>(&line_string)?;
                        true
                    }
                    other => wrong_kind("linestring", other, ignore_per_row_errors)?,
                },
            };
            validity.push(is_valid);
            rows.push(coords.len());
        }
        Ok(LineStringArray::from_parts(
            List::from_offsets(rows.finish(), validity.finish()),
            coords.finish(),
        ))
    }

    /// See [`GeosGeometries::into_line_string_array`].
    /// GEOS gives a multilinestring of one part as a plain linestring.
    /// So a linestring is read as a single part, or as none if it is empty.
    /// A geometry that is not a linestring or multilinestring fails the same way.
    pub fn into_multi_line_string_array<const Z: bool>(
        self,
        ignore_per_row_errors: bool,
    ) -> GeoComputeResult<MultiLineStringArray> {
        let mut validity = ValidityBuilder::with_capacity(self.geometries.len());
        let mut coords = CoordsBuilder::with_capacity(Z, self.size_hint.coords);
        let mut lines = OffsetsBuilder::with_capacity(self.size_hint.parts[0]);
        let mut rows = OffsetsBuilder::with_capacity(self.geometries.len());
        for geometry in self.geometries {
            let is_valid = match geometry {
                GeosRow::Null => false,
                GeosRow::Empty => true,
                GeosRow::Geometry(geometry) => match geometry.geometry_type()? {
                    GeometryTypes::LineString if geometry.is_empty()? => true,
                    GeometryTypes::LineString => {
                        coords.push::<Z, false>(&geometry)?;
                        lines.push(coords.len());
                        true
                    }
                    GeometryTypes::MultiLineString => {
                        for n in 0..geometry.get_num_geometries()? {
                            coords.push::<Z, false>(&geometry.get_geometry_n(n)?)?;
                            lines.push(coords.len());
                        }
                        true
                    }
                    other => wrong_kind("multilinestring", other, ignore_per_row_errors)?,
                },
            };
            validity.push(is_valid);
            rows.push(lines.len());
        }
        Ok(MultiLineStringArray::from_parts(
            List::from_offsets(rows.finish(), validity.finish()),
            ValidList::from_offsets(lines.finish()),
            coords.finish(),
        ))
    }
}
