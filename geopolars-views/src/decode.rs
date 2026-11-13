//! Building our native geometries from geometries decoded by another crate (`wkb`, `wkt`),
//! which hands them over through geo-traits.

use geo_traits::{
    CoordTrait, Dimensions, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::offset::Offsets;
use polars_core::prelude::{polars_bail, polars_ensure, CompatLevel, PolarsResult};

use crate::{GeoDimension, Kind};

/// Builds the storage of one geometry column, one decoded geometry at a time.
///
/// `offsets[level]` holds the offsets of list level `level`, outermost first,
/// each pointing into the level below it (the coordinates, for the innermost one).
pub struct GeometryBuilder {
    kind: Kind,
    dim: GeoDimension,
    label: &'static str,
    /// One buffer per coordinate, in [`GeoDimension::field_names`] order.
    coords: Vec<Vec<f64>>,
    offsets: Vec<Vec<i64>>,
    /// Only the outermost level can be null.
    validity: MutableBitmap,
}

impl GeometryBuilder {
    /// `label` is how the input is spelled in an error message: `WKB`.
    pub fn new(kind: Kind, dim: GeoDimension, label: &'static str, capacity: usize) -> Self {
        Self {
            kind,
            dim,
            label,
            coords: vec![Vec::with_capacity(capacity); dim.field_names().len()],
            offsets: (0..kind.nesting()).map(|_| vec![0]).collect(),
            validity: MutableBitmap::with_capacity(capacity),
        }
    }

    fn coord(&mut self, coord: &impl CoordTrait<T = f64>) {
        for (n, values) in self.coords.iter_mut().enumerate() {
            values.push(coord.nth_or_panic(n));
        }
    }

    /// GeoArrow (like WKB) spells an empty point as all-NaN coordinates.
    fn empty_coord(&mut self) {
        for values in &mut self.coords {
            values.push(f64::NAN);
        }
    }

    /// Ends the current entry of list level `level`.
    fn close(&mut self, level: usize) {
        let len = match self.offsets.get(level + 1) {
            Some(below) => below.len() - 1,
            None => self.coords[0].len(),
        };
        self.offsets[level].push(len as i64);
    }

    fn point(&mut self, point: &impl PointTrait<T = f64>) {
        match point.coord() {
            Some(coord) => self.coord(&coord),
            None => self.empty_coord(),
        }
    }

    fn line(&mut self, level: usize, line: &impl LineStringTrait<T = f64>) {
        for coord in line.coords() {
            self.coord(&coord);
        }
        self.close(level);
    }

    fn polygon(&mut self, level: usize, polygon: &impl PolygonTrait<T = f64>) {
        if let Some(exterior) = polygon.exterior() {
            self.line(level + 1, &exterior);
        }
        for interior in polygon.interiors() {
            self.line(level + 1, &interior);
        }
        self.close(level);
    }

    pub fn null(&mut self) {
        match self.kind {
            Kind::Point => self.empty_coord(),
            _ => self.close(0),
        }
        self.validity.push(false);
    }

    /// Decodes one geometry. A single geometry is promoted into the multi geometry
    /// of the same type, as GeoParquet files commonly mix those two.
    pub fn geometry(&mut self, geometry: &impl GeometryTrait<T = f64>) -> PolarsResult<()> {
        polars_ensure!(
            geometry.dim() == Dimensions::from(self.dim),
            ComputeError: "cannot decode {} with {} coordinates into a `{}` column with {} coordinates",
            self.label, dimension_name(geometry.dim()), self.kind.name(), self.dim.tag()
        );

        match (self.kind, geometry.as_type()) {
            (Kind::Point, GeometryType::Point(point)) => self.point(point),
            (Kind::LineString, GeometryType::LineString(line)) => self.line(0, line),
            (Kind::Polygon, GeometryType::Polygon(polygon)) => self.polygon(0, polygon),
            (Kind::MultiPoint, GeometryType::MultiPoint(points)) => {
                for point in points.points() {
                    self.point(&point);
                }
                self.close(0);
            }
            (Kind::MultiPoint, GeometryType::Point(point)) => {
                // An empty point promotes into an empty multipoint, not into one NaN point.
                if let Some(coord) = point.coord() {
                    self.coord(&coord);
                }
                self.close(0);
            }
            (Kind::MultiLineString, GeometryType::MultiLineString(lines)) => {
                for line in lines.line_strings() {
                    self.line(1, &line);
                }
                self.close(0);
            }
            (Kind::MultiLineString, GeometryType::LineString(line)) => {
                self.line(1, line);
                self.close(0);
            }
            (Kind::MultiPolygon, GeometryType::MultiPolygon(polygons)) => {
                for polygon in polygons.polygons() {
                    self.polygon(1, &polygon);
                }
                self.close(0);
            }
            (Kind::MultiPolygon, GeometryType::Polygon(polygon)) => {
                self.polygon(1, polygon);
                self.close(0);
            }
            (_, other) => polars_bail!(
                ComputeError: "cannot decode a {} {} into a `{}` column",
                self.label, type_name(&other), self.kind.name()
            ),
        }
        self.validity.push(true);
        Ok(())
    }

    pub fn finish(self) -> PolarsResult<Box<dyn Array>> {
        let validity: Bitmap = self.validity.into();
        let validity = (validity.unset_bits() > 0).then_some(validity);
        let len = self.coords[0].len();

        let coords_dtype = self.dim.coordinates().to_arrow(CompatLevel::newest());
        let values = self
            .coords
            .into_iter()
            .map(|values| PrimitiveArray::from_vec(values).boxed())
            .collect();
        let mut array: Box<dyn Array> = if self.offsets.is_empty() {
            StructArray::new(coords_dtype, len, values, validity.clone()).boxed()
        } else {
            StructArray::new(coords_dtype, len, values, None).boxed()
        };

        let levels = self.offsets.len();
        for (level, offsets) in self.offsets.into_iter().enumerate().rev() {
            let dtype = ListArray::<i64>::default_datatype(array.dtype().clone());
            let offsets = Offsets::try_from(offsets)?.into();
            let validity = (level == 0 && levels > 0)
                .then(|| validity.clone())
                .flatten();
            array = ListArray::<i64>::new(dtype, offsets, array, validity).boxed();
        }
        Ok(array)
    }
}

/// How a decoded dimension is spelled in an error message: `xyz`.
fn dimension_name(dim: Dimensions) -> String {
    match dim {
        Dimensions::Xy => "xy".into(),
        Dimensions::Xyz => "xyz".into(),
        Dimensions::Xym => "xym".into(),
        Dimensions::Xyzm => "xyzm".into(),
        Dimensions::Unknown(n) => format!("{n}"),
    }
}

/// How a decoded geometry is spelled in an error message: `MultiPoint`.
fn type_name<P, L, Y, MP, ML, MY, GC, R, T, LN>(
    geometry: &GeometryType<'_, P, L, Y, MP, ML, MY, GC, R, T, LN>,
) -> &'static str
where
    P: PointTrait,
    L: LineStringTrait,
    Y: PolygonTrait,
    MP: MultiPointTrait,
    ML: MultiLineStringTrait,
    MY: MultiPolygonTrait,
    GC: geo_traits::GeometryCollectionTrait,
    R: geo_traits::RectTrait,
    T: geo_traits::TriangleTrait,
    LN: geo_traits::LineTrait,
{
    match geometry {
        GeometryType::Point(_) => "Point",
        GeometryType::LineString(_) => "LineString",
        GeometryType::Polygon(_) => "Polygon",
        GeometryType::MultiPoint(_) => "MultiPoint",
        GeometryType::MultiLineString(_) => "MultiLineString",
        GeometryType::MultiPolygon(_) => "MultiPolygon",
        GeometryType::GeometryCollection(_) => "GeometryCollection",
        GeometryType::Rect(_) => "Rect",
        GeometryType::Triangle(_) => "Triangle",
        GeometryType::Line(_) => "Line",
    }
}
