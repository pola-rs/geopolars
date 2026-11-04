use geopolars_core::{Coords, GeoDimension, PointArray, ValidCoords};
use geos::{Geom, GeometryTypes};

use super::{coord_type, GeosGeometries, GeosRow, ValidityBuilder};
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
}
