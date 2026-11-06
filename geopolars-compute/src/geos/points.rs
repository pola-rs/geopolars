use geopolars_core::{Coords, GeoDimension, List, MultiPointArray, PointArray, ValidCoords};
use geos::{Geom, GeometryTypes};

use super::{
    coord_type, wrong_kind, CoordsBuilder, GeosGeometries, GeosRow, OffsetsBuilder, ValidityBuilder,
};
use crate::GeoComputeResult;

impl GeosGeometries {
    /// Every GEOS geometry is freed once it is copied.
    /// An empty row or point gives an empty point (NaN on every axis).
    /// A geometry that is not a point is an error.
    pub fn into_point_array(self, dim: GeoDimension) -> GeoComputeResult<PointArray> {
        let rows = self.geometries.len();
        let width = dim.field_names().len();
        let empty = [f64::NAN; 4];
        let mut validity = ValidityBuilder::with_capacity(rows);
        let mut axes: Vec<Vec<f64>> = (0..width).map(|_| Vec::with_capacity(rows)).collect();
        let mut push = |coord: &[f64]| {
            for (axis, &value) in axes.iter_mut().zip(coord) {
                axis.push(value);
            }
        };
        for geometry in self.geometries {
            let is_valid = match geometry {
                GeosRow::Null => false,
                GeosRow::Empty => {
                    push(&empty);
                    true
                }
                GeosRow::Geometry(point) => match point.geometry_type()? {
                    GeometryTypes::Point if point.is_empty()? => {
                        push(&empty);
                        true
                    }
                    GeometryTypes::Point => {
                        push(&point.get_coord_seq()?.as_buffer(Some(coord_type(dim)))?);
                        true
                    }
                    other => {
                        let message = format!("expected a point, got a {other:?}");
                        return Err(geos::Error::ConversionError(message).into());
                    }
                },
            };
            if !is_valid {
                // The value under a null row is never read.
                push(&empty);
            }
            validity.push(is_valid);
        }
        let mut axes = axes.into_iter();
        let mut next = || axes.next().expect("one axis per field of `dim`");
        let (x, y) = (next(), next());
        let axes = match dim {
            GeoDimension::XY => (x, y, None, None),
            GeoDimension::XYZ => (x, y, Some(next()), None),
            GeoDimension::XYM => (x, y, None, Some(next())),
            GeoDimension::XYZM => {
                let z = next();
                (x, y, Some(z), Some(next()))
            }
        };
        Ok(PointArray::new(Coords::from_valid(
            ValidCoords::from_axes(axes),
            validity.finish(),
        )))
    }

    /// Our multipoint array of the geometries, one row each.
    /// Every GEOS geometry is freed once it is copied.
    /// GEOS gives a multipoint of one part as a plain point.
    /// So a point is read as a single part, or as none if it is empty.
    /// A geometry that is not a point or multipoint is an error.
    /// If `ignore_per_row_errors=true`, its row is null instead.
    pub fn into_multi_point_array<const Z: bool>(
        self,
        ignore_per_row_errors: bool,
    ) -> GeoComputeResult<MultiPointArray> {
        let mut validity = ValidityBuilder::with_capacity(self.geometries.len());
        let mut coords = CoordsBuilder::with_capacity(Z, self.size_hint.coords);
        let mut rows = OffsetsBuilder::with_capacity(self.geometries.len());
        for geometry in self.geometries {
            let is_valid = match geometry {
                GeosRow::Null => false,
                GeosRow::Empty => true,
                GeosRow::Geometry(geometry) => match geometry.geometry_type()? {
                    // An empty point has no coordinates, so it adds no part.
                    GeometryTypes::Point => {
                        coords.push::<Z, false>(&geometry)?;
                        true
                    }
                    GeometryTypes::MultiPoint => {
                        for n in 0..geometry.get_num_geometries()? {
                            coords.push::<Z, false>(&geometry.get_geometry_n(n)?)?;
                        }
                        true
                    }
                    other => wrong_kind("multipoint", other, ignore_per_row_errors)?,
                },
            };
            validity.push(is_valid);
            rows.push(coords.len());
        }
        Ok(MultiPointArray::from_parts(
            List::from_offsets(rows.finish(), validity.finish()),
            coords.finish(),
        ))
    }
}
