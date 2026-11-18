use geopolars_views::{Coords, GeoDimension, List, MultiPointArray, PointArray, ValidCoords};
use geos::{Geom, GeometryTypes};

use super::{
    coord_type, wrong_kind, CoordsBuilder, GeosGeometries, GeosRow, GeosRows, OffsetsBuilder,
};
use crate::{GeoComputeResult, OnRowError};

impl GeosGeometries {
    /// Every GEOS geometry is freed once it is copied.
    /// An empty row or point gives an empty point (NaN on every axis).
    /// A geometry that is not a point is an error.
    pub fn into_point_array(self, dim: GeoDimension) -> GeoComputeResult<PointArray> {
        let GeosRows { rows, validity } = self.geometries;
        let width = dim.field_names().len();
        // Also what a null row gets, as it holds `Empty`.
        let empty = [f64::NAN; 4];
        let mut axes: Vec<Vec<f64>> = (0..width).map(|_| Vec::with_capacity(rows.len())).collect();
        let mut push = |coord: &[f64]| {
            for (axis, &value) in axes.iter_mut().zip(coord) {
                axis.push(value);
            }
        };
        for geometry in rows {
            match geometry {
                GeosRow::Empty => push(&empty),
                GeosRow::Geometry {
                    geometry: point, ..
                } => match point.geometry_type()? {
                    GeometryTypes::Point if point.is_empty()? => push(&empty),
                    GeometryTypes::Point => {
                        push(&point.get_coord_seq()?.as_buffer(Some(coord_type(dim)))?);
                    }
                    other => {
                        let message = format!("expected a point, got a {other:?}");
                        return Err(geos::Error::ConversionError(message).into());
                    }
                },
            }
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
    /// As GEOS gives a multipoint of one part as a plain point,
    /// a point is read as a single part, or as none if it is empty.
    /// A geometry that is not a point or multipoint goes to `on_row_error`.
    pub fn into_multi_point_array<const Z: bool>(
        self,
        on_row_error: OnRowError,
    ) -> GeoComputeResult<MultiPointArray> {
        let GeosRows {
            rows: geometries,
            mut validity,
        } = self.geometries;
        let mut coords = CoordsBuilder::with_capacity(Z, self.size_hint.coords);
        let mut rows = OffsetsBuilder::with_capacity(geometries.len());
        for (i, geometry) in geometries.into_iter().enumerate() {
            let is_valid = match geometry {
                GeosRow::Empty => true,
                GeosRow::Geometry { geometry, .. } => {
                    match geometry.geometry_type()? {
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
                        other => wrong_kind("multipoint", other, on_row_error)?,
                    }
                }
            };
            if !is_valid {
                validity.set_null(i);
            }
            rows.push(coords.len());
        }
        Ok(MultiPointArray::from_parts(
            List::from_offsets(rows.finish(), validity.finish()),
            coords.finish(),
        ))
    }
}
