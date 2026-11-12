//! [geo-traits](geo_traits) for the row types.
//!
//! These exist because the `wkb` and `wkt` writers only take geometries through geo-traits.
//! geo-traits *also* has **no null** interior coordinates, so everything below a row is a [ValidCoords].
//!
//! Don't shoot me for using lifetimes here!
//! Owning their data would clone the column's buffers each time:
//! an atomic refcount per row, and per coordinate for the writers.

use geo_traits::{
    CoordTrait, Dimensions, GeometryTrait, LineStringTrait, MultiLineStringTrait, MultiPointTrait,
    MultiPolygonTrait, PointTrait, PolygonTrait, UnimplementedGeometryCollection,
    UnimplementedLine, UnimplementedLineString, UnimplementedMultiLineString,
    UnimplementedMultiPoint, UnimplementedMultiPolygon, UnimplementedPolygon, UnimplementedRect,
    UnimplementedTriangle,
};

use crate::{
    GeoDimension, ScalarLineString, ScalarMultiLineString, ScalarMultiPoint, ScalarMultiPolygon,
    ScalarPoint, ScalarPolygon, ValidCoords,
};

/// One coordinate, by value.
/// Also the point of a multipoint, which is never empty.
#[derive(Clone, Copy)]
pub struct Coord {
    /// In storage order: x, y, then z and/or m. Only the first ones of `dim` are set.
    values: [f64; 4],
    dim: GeoDimension,
}

impl Coord {
    /// Coordinate `i` of `coords`.
    fn new(coords: &ValidCoords, i: usize) -> Self {
        let dim = coords.dim();
        let mut values = [0.0; 4];
        for (n, value) in values[..dim.field_names().len()].iter_mut().enumerate() {
            *value = coords.nth(i, n);
        }
        Self { values, dim }
    }
}

impl CoordTrait for Coord {
    type T = f64;

    fn dim(&self) -> Dimensions {
        self.dim.into()
    }

    fn x(&self) -> f64 {
        self.values[0]
    }

    fn y(&self) -> f64 {
        self.values[1]
    }

    fn nth_or_panic(&self, n: usize) -> f64 {
        self.values[..self.dim.field_names().len()][n]
    }
}

impl PointTrait for Coord {
    type CoordType<'b>
        = Coord
    where
        Self: 'b;

    fn coord(&self) -> Option<Coord> {
        Some(*self)
    }
}

impl GeometryTrait for Coord {
    type T = f64;
    type PointType<'b>
        = Coord
    where
        Self: 'b;
    type LineStringType<'b>
        = UnimplementedLineString<f64>
    where
        Self: 'b;
    type PolygonType<'b>
        = UnimplementedPolygon<f64>
    where
        Self: 'b;
    type MultiPointType<'b>
        = UnimplementedMultiPoint<f64>
    where
        Self: 'b;
    type MultiLineStringType<'b>
        = UnimplementedMultiLineString<f64>
    where
        Self: 'b;
    type MultiPolygonType<'b>
        = UnimplementedMultiPolygon<f64>
    where
        Self: 'b;
    type GeometryCollectionType<'b>
        = UnimplementedGeometryCollection<f64>
    where
        Self: 'b;
    type RectType<'b>
        = UnimplementedRect<f64>
    where
        Self: 'b;
    type TriangleType<'b>
        = UnimplementedTriangle<f64>
    where
        Self: 'b;
    type LineType<'b>
        = UnimplementedLine<f64>
    where
        Self: 'b;

    fn dim(&self) -> Dimensions {
        self.dim.into()
    }

    fn as_type(
        &self,
    ) -> geo_traits::GeometryType<
        '_,
        Coord,
        UnimplementedLineString<f64>,
        UnimplementedPolygon<f64>,
        UnimplementedMultiPoint<f64>,
        UnimplementedMultiLineString<f64>,
        UnimplementedMultiPolygon<f64>,
        UnimplementedGeometryCollection<f64>,
        UnimplementedRect<f64>,
        UnimplementedTriangle<f64>,
        UnimplementedLine<f64>,
    > {
        geo_traits::GeometryType::Point(self)
    }
}

/// A point of its own is empty where x and y are NaN, as GeoArrow spells it.
impl PointTrait for ScalarPoint<'_> {
    type CoordType<'b>
        = Coord
    where
        Self: 'b;

    fn coord(&self) -> Option<Coord> {
        let coords = self.coords.ignoring_validity();
        let empty = coords.nth(self.index, 0).is_nan() && coords.nth(self.index, 1).is_nan();
        (!empty).then(|| Coord::new(coords, self.index))
    }
}

impl LineStringTrait for ScalarLineString<'_> {
    type CoordType<'b>
        = Coord
    where
        Self: 'b;

    fn num_coords(&self) -> usize {
        self.vertices.len()
    }

    unsafe fn coord_unchecked(&self, i: usize) -> Coord {
        Coord::new(self.coords, self.vertices.start + i)
    }
}

impl<'a> ScalarPolygon<'a> {
    fn ring(&self, i: usize) -> ScalarLineString<'a> {
        ScalarLineString {
            coords: self.coords,
            vertices: self.rings.range(self.parts.start + i),
        }
    }
}

impl<'a> PolygonTrait for ScalarPolygon<'a> {
    type RingType<'b>
        = ScalarLineString<'a>
    where
        Self: 'b;

    fn exterior(&self) -> Option<ScalarLineString<'a>> {
        (!self.parts.is_empty()).then(|| self.ring(0))
    }

    fn num_interiors(&self) -> usize {
        self.parts.len().saturating_sub(1)
    }

    unsafe fn interior_unchecked(&self, i: usize) -> ScalarLineString<'a> {
        self.ring(i + 1)
    }
}

impl MultiPointTrait for ScalarMultiPoint<'_> {
    type InnerPointType<'b>
        = Coord
    where
        Self: 'b;

    fn num_points(&self) -> usize {
        self.vertices.len()
    }

    unsafe fn point_unchecked(&self, i: usize) -> Coord {
        Coord::new(self.coords, self.vertices.start + i)
    }
}

impl<'a> MultiLineStringTrait for ScalarMultiLineString<'a> {
    type InnerLineStringType<'b>
        = ScalarLineString<'a>
    where
        Self: 'b;

    fn num_line_strings(&self) -> usize {
        self.parts.len()
    }

    unsafe fn line_string_unchecked(&self, i: usize) -> ScalarLineString<'a> {
        ScalarLineString {
            coords: self.coords,
            vertices: self.lines.range(self.parts.start + i),
        }
    }
}

impl<'a> MultiPolygonTrait for ScalarMultiPolygon<'a> {
    type InnerPolygonType<'b>
        = ScalarPolygon<'a>
    where
        Self: 'b;

    fn num_polygons(&self) -> usize {
        self.parts.len()
    }

    unsafe fn polygon_unchecked(&self, i: usize) -> ScalarPolygon<'a> {
        ScalarPolygon {
            rings: self.rings,
            coords: self.coords,
            parts: self.polygons.range(self.parts.start + i),
        }
    }
}

/// Every row type is a geometry of the one variant it is.
macro_rules! impl_geometry {
    ($($ty:ident => $variant:ident),*) => {$(
        impl<'a> GeometryTrait for $ty<'a> {
            type T = f64;
            type PointType<'b> = ScalarPoint<'a> where Self: 'b;
            type LineStringType<'b> = ScalarLineString<'a> where Self: 'b;
            type PolygonType<'b> = ScalarPolygon<'a> where Self: 'b;
            type MultiPointType<'b> = ScalarMultiPoint<'a> where Self: 'b;
            type MultiLineStringType<'b> = ScalarMultiLineString<'a> where Self: 'b;
            type MultiPolygonType<'b> = ScalarMultiPolygon<'a> where Self: 'b;
            type GeometryCollectionType<'b> = UnimplementedGeometryCollection<f64> where Self: 'b;
            type RectType<'b> = UnimplementedRect<f64> where Self: 'b;
            type TriangleType<'b> = UnimplementedTriangle<f64> where Self: 'b;
            type LineType<'b> = UnimplementedLine<f64> where Self: 'b;

            fn dim(&self) -> Dimensions {
                self.coords.dim().into()
            }

            fn as_type(
                &self,
            ) -> geo_traits::GeometryType<
                '_,
                ScalarPoint<'a>,
                ScalarLineString<'a>,
                ScalarPolygon<'a>,
                ScalarMultiPoint<'a>,
                ScalarMultiLineString<'a>,
                ScalarMultiPolygon<'a>,
                UnimplementedGeometryCollection<f64>,
                UnimplementedRect<f64>,
                UnimplementedTriangle<f64>,
                UnimplementedLine<f64>,
            > {
                geo_traits::GeometryType::$variant(self)
            }
        }
    )*};
}

impl_geometry!(
    ScalarPoint => Point,
    ScalarLineString => LineString,
    ScalarPolygon => Polygon,
    ScalarMultiPoint => MultiPoint,
    ScalarMultiLineString => MultiLineString,
    ScalarMultiPolygon => MultiPolygon
);

/// Any one of the row types.
pub enum Geometry<'a> {
    Point(ScalarPoint<'a>),
    LineString(ScalarLineString<'a>),
    Polygon(ScalarPolygon<'a>),
    MultiPoint(ScalarMultiPoint<'a>),
    MultiLineString(ScalarMultiLineString<'a>),
    MultiPolygon(ScalarMultiPolygon<'a>),
}

impl<'a> GeometryTrait for Geometry<'a> {
    type T = f64;
    type PointType<'b>
        = ScalarPoint<'a>
    where
        Self: 'b;
    type LineStringType<'b>
        = ScalarLineString<'a>
    where
        Self: 'b;
    type PolygonType<'b>
        = ScalarPolygon<'a>
    where
        Self: 'b;
    type MultiPointType<'b>
        = ScalarMultiPoint<'a>
    where
        Self: 'b;
    type MultiLineStringType<'b>
        = ScalarMultiLineString<'a>
    where
        Self: 'b;
    type MultiPolygonType<'b>
        = ScalarMultiPolygon<'a>
    where
        Self: 'b;
    type GeometryCollectionType<'b>
        = UnimplementedGeometryCollection<f64>
    where
        Self: 'b;
    type RectType<'b>
        = UnimplementedRect<f64>
    where
        Self: 'b;
    type TriangleType<'b>
        = UnimplementedTriangle<f64>
    where
        Self: 'b;
    type LineType<'b>
        = UnimplementedLine<f64>
    where
        Self: 'b;

    fn dim(&self) -> Dimensions {
        match self {
            Geometry::Point(g) => g.dim(),
            Geometry::LineString(g) => g.dim(),
            Geometry::Polygon(g) => g.dim(),
            Geometry::MultiPoint(g) => g.dim(),
            Geometry::MultiLineString(g) => g.dim(),
            Geometry::MultiPolygon(g) => g.dim(),
        }
    }

    fn as_type(
        &self,
    ) -> geo_traits::GeometryType<
        '_,
        ScalarPoint<'a>,
        ScalarLineString<'a>,
        ScalarPolygon<'a>,
        ScalarMultiPoint<'a>,
        ScalarMultiLineString<'a>,
        ScalarMultiPolygon<'a>,
        UnimplementedGeometryCollection<f64>,
        UnimplementedRect<f64>,
        UnimplementedTriangle<f64>,
        UnimplementedLine<f64>,
    > {
        match self {
            Geometry::Point(g) => geo_traits::GeometryType::Point(g),
            Geometry::LineString(g) => geo_traits::GeometryType::LineString(g),
            Geometry::Polygon(g) => geo_traits::GeometryType::Polygon(g),
            Geometry::MultiPoint(g) => geo_traits::GeometryType::MultiPoint(g),
            Geometry::MultiLineString(g) => geo_traits::GeometryType::MultiLineString(g),
            Geometry::MultiPolygon(g) => geo_traits::GeometryType::MultiPolygon(g),
        }
    }
}
