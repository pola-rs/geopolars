use polars_arrow::bitmap::Bitmap;

use crate::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray,
};

/// An array of any one kind of geometry, for kernels that take every kind.
pub enum GeometryArray {
    Point(PointArray),
    LineString(LineStringArray),
    Polygon(PolygonArray),
    MultiPoint(MultiPointArray),
    MultiLineString(MultiLineStringArray),
    MultiPolygon(MultiPolygonArray),
}

impl GeometryArray {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            GeometryArray::Point(array) => array.coords.len(),
            GeometryArray::LineString(array) => array.rows.len(),
            GeometryArray::Polygon(array) => array.rows.len(),
            GeometryArray::MultiPoint(array) => array.rows.len(),
            GeometryArray::MultiLineString(array) => array.rows.len(),
            GeometryArray::MultiPolygon(array) => array.rows.len(),
        }
    }

    pub fn validity(&self) -> Option<&Bitmap> {
        match self {
            GeometryArray::Point(array) => array.coords.validity(),
            GeometryArray::LineString(array) => array.rows.validity(),
            GeometryArray::Polygon(array) => array.rows.validity(),
            GeometryArray::MultiPoint(array) => array.rows.validity(),
            GeometryArray::MultiLineString(array) => array.rows.validity(),
            GeometryArray::MultiPolygon(array) => array.rows.validity(),
        }
    }

    pub fn with_validity(self, validity: Option<Bitmap>) -> Self {
        match self {
            GeometryArray::Point(mut array) => {
                array.coords = array.coords.with_validity(validity);
                GeometryArray::Point(array)
            }
            GeometryArray::LineString(mut array) => {
                array.rows = array.rows.with_validity(validity);
                GeometryArray::LineString(array)
            }
            GeometryArray::Polygon(mut array) => {
                array.rows = array.rows.with_validity(validity);
                GeometryArray::Polygon(array)
            }
            GeometryArray::MultiPoint(mut array) => {
                array.rows = array.rows.with_validity(validity);
                GeometryArray::MultiPoint(array)
            }
            GeometryArray::MultiLineString(mut array) => {
                array.rows = array.rows.with_validity(validity);
                GeometryArray::MultiLineString(array)
            }
            GeometryArray::MultiPolygon(mut array) => {
                array.rows = array.rows.with_validity(validity);
                GeometryArray::MultiPolygon(array)
            }
        }
    }
}
