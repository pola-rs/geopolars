use polars_arrow::array::{Array, ListArray};
use polars_arrow::bitmap::Bitmap;
use polars_core::prelude::{CompatLevel, PolarsResult};

use crate::{
    downcast, Coords, GeoDimension, Geometry, Kind, LineStringArray, MultiLineStringArray,
    MultiPointArray, MultiPolygonArray, PointArray, PolygonArray,
};

/// An array of any one kind of geometry,
/// for kernels that take every kind, and for the geometries `geopolars-core` hands back.
pub enum GeometryArray {
    Point(PointArray),
    LineString(LineStringArray),
    Polygon(PolygonArray),
    MultiPoint(MultiPointArray),
    MultiLineString(MultiLineStringArray),
    MultiPolygon(MultiPolygonArray),
}

impl GeometryArray {
    /// Wraps a chunk of a `kind` column's storage.
    pub fn new(chunk: &dyn Array, kind: Kind) -> PolarsResult<Self> {
        let rows = |chunk| downcast::<ListArray<i64>>(chunk, "a list of parts");
        Ok(match kind {
            Kind::Point => GeometryArray::Point(PointArray::new(Coords::new(chunk)?)),
            Kind::MultiPoint => GeometryArray::MultiPoint(MultiPointArray::new(rows(chunk)?)?),
            Kind::LineString => GeometryArray::LineString(LineStringArray::new(rows(chunk)?)?),
            Kind::MultiLineString => {
                GeometryArray::MultiLineString(MultiLineStringArray::new(rows(chunk)?)?)
            }
            Kind::Polygon => GeometryArray::Polygon(PolygonArray::new(rows(chunk)?)?),
            Kind::MultiPolygon => {
                GeometryArray::MultiPolygon(MultiPolygonArray::new(rows(chunk)?)?)
            }
        })
    }

    pub fn kind(&self) -> Kind {
        match self {
            GeometryArray::Point(_) => Kind::Point,
            GeometryArray::LineString(_) => Kind::LineString,
            GeometryArray::Polygon(_) => Kind::Polygon,
            GeometryArray::MultiPoint(_) => Kind::MultiPoint,
            GeometryArray::MultiLineString(_) => Kind::MultiLineString,
            GeometryArray::MultiPolygon(_) => Kind::MultiPolygon,
        }
    }

    pub fn dim(&self) -> GeoDimension {
        match self {
            GeometryArray::Point(array) => array.coords.dim(),
            GeometryArray::LineString(array) => array.coords.dim(),
            GeometryArray::Polygon(array) => array.coords.dim(),
            GeometryArray::MultiPoint(array) => array.coords.dim(),
            GeometryArray::MultiLineString(array) => array.coords.dim(),
            GeometryArray::MultiPolygon(array) => array.coords.dim(),
        }
    }

    /// The polars storage of the column, taking over its buffers:
    /// always the storage [`GeoDimension::storage`] gives for [`dim`](Self::dim)
    /// and the [`nesting`](Kind::nesting) of [`kind`](Self::kind).
    pub fn into_storage(self) -> Box<dyn Array> {
        let (kind, dim) = (self.kind(), self.dim());
        let storage = match self {
            GeometryArray::Point(array) => array.into_struct_array().boxed(),
            GeometryArray::LineString(array) => array.into_list_array().boxed(),
            GeometryArray::Polygon(array) => array.into_list_array().boxed(),
            GeometryArray::MultiPoint(array) => array.into_list_array().boxed(),
            GeometryArray::MultiLineString(array) => array.into_list_array().boxed(),
            GeometryArray::MultiPolygon(array) => array.into_list_array().boxed(),
        };
        debug_assert_eq!(
            storage.dtype(),
            &dim.storage(kind.nesting()).to_arrow(CompatLevel::newest()),
            "the view of a `{}` gave another storage",
            kind.display()
        );
        storage
    }

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

    /// Calls `f` with every row: the geometry, or `None` for a null.
    /// For the `wkb` and `wkt` writers, which take geometries through geo-traits.
    pub fn for_each_geometry<'a>(
        &'a self,
        mut f: impl FnMut(Option<Geometry<'a>>) -> PolarsResult<()>,
    ) -> PolarsResult<()> {
        match self {
            GeometryArray::Point(array) => {
                for_each_row(self, &mut f, |i| Geometry::Point(array.get(i)))
            }
            GeometryArray::LineString(array) => {
                for_each_row(self, &mut f, |i| Geometry::LineString(array.get(i)))
            }
            GeometryArray::Polygon(array) => {
                for_each_row(self, &mut f, |i| Geometry::Polygon(array.get(i)))
            }
            GeometryArray::MultiPoint(array) => {
                for_each_row(self, &mut f, |i| Geometry::MultiPoint(array.get(i)))
            }
            GeometryArray::MultiLineString(array) => {
                for_each_row(self, &mut f, |i| Geometry::MultiLineString(array.get(i)))
            }
            GeometryArray::MultiPolygon(array) => {
                for_each_row(self, &mut f, |i| Geometry::MultiPolygon(array.get(i)))
            }
        }
    }
}

/// Calls `f` with `row(i)` for every row `i` of `array`, or `None` where it is null.
fn for_each_row<'a>(
    array: &GeometryArray,
    f: &mut impl FnMut(Option<Geometry<'a>>) -> PolarsResult<()>,
    row: impl Fn(usize) -> Geometry<'a>,
) -> PolarsResult<()> {
    let validity = array.validity();
    (0..array.len()).try_for_each(|i| f(validity.is_none_or(|v| v.get_bit(i)).then(|| row(i))))
}
