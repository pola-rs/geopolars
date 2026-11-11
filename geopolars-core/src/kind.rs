//! GeoArrow geometries are not implemented as concrete rust types,
//! but as enums ([Kind]) so that dispatching follows the way Polars does it.

/// A GeoArrow native geometry, which takes some [`GeoDimension`](crate::GeoDimension).
/// Used for dispatching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
}

impl Kind {
    pub const ALL: [Kind; 6] = [
        Kind::Point,
        Kind::LineString,
        Kind::Polygon,
        Kind::MultiPoint,
        Kind::MultiLineString,
        Kind::MultiPolygon,
    ];

    /// The `ARROW:extension:name` this geometry is registered under.
    /// The Python side must spell the same name.
    pub const fn name(self) -> &'static str {
        match self {
            Kind::Point => "geoarrow.point",
            Kind::LineString => "geoarrow.linestring",
            Kind::Polygon => "geoarrow.polygon",
            Kind::MultiPoint => "geoarrow.multipoint",
            Kind::MultiLineString => "geoarrow.multilinestring",
            Kind::MultiPolygon => "geoarrow.multipolygon",
        }
    }

    /// How the geometry is shown in a DataFrame header: `point[xy]`.
    /// Should not be confused with [std::fmt::Display]'s `display`!
    pub const fn display(self) -> &'static str {
        match self {
            Kind::Point => "point",
            Kind::LineString => "linestring",
            Kind::Polygon => "polygon",
            Kind::MultiPoint => "multipoint",
            Kind::MultiLineString => "multilinestring",
            Kind::MultiPolygon => "multipolygon",
        }
    }

    /// How the geometry is spelled as a type: `PointXY`, `LineStringXY`.
    pub const fn type_name(self) -> &'static str {
        match self {
            Kind::Point => "Point",
            Kind::LineString => "LineString",
            Kind::Polygon => "Polygon",
            Kind::MultiPoint => "MultiPoint",
            Kind::MultiLineString => "MultiLineString",
            Kind::MultiPolygon => "MultiPolygon",
        }
    }

    pub const fn is_multi(self) -> bool {
        matches!(
            self,
            Kind::MultiPoint | Kind::MultiLineString | Kind::MultiPolygon
        )
    }

    /// Tells us how deeply nested the data type is.
    /// A geoarrow.linestring is a collection of points, so it is 1.
    /// A geoarrow.polygon is a collection of linestings, so it is 2.
    /// A geoarrow.multipolygon is a collection of polygons, so it is 3.
    ///
    /// Warning: This does not uniquely identify a geometry kind:
    /// E.g. geoarrow.linestring and geoarrow.multipoint both have a nesting of 1.
    /// The extension name is what tells them apart,
    /// which is why a column's kind is read from its name and the nesting derived from it,
    /// rather than the other way round.
    pub const fn nesting(self) -> u8 {
        match self {
            Kind::Point => 0,
            Kind::LineString | Kind::MultiPoint => 1,
            Kind::Polygon | Kind::MultiLineString => 2,
            Kind::MultiPolygon => 3,
        }
    }

    pub fn from_name(name: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|kind| kind.name() == name)
    }

    /// Inverse of [`Kind::display`].
    pub fn from_display(display: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|kind| kind.display() == display)
    }

    /// The names, for an error message:
    /// e.g. `geoarrow.point` or `geoarrow.linestring`.
    pub fn names() -> String {
        let names: Vec<String> = Kind::ALL
            .iter()
            .map(|kind| format!("`{}`", kind.name()))
            .collect();
        match names.split_last() {
            Some((last, [])) => last.clone(),
            Some((last, rest)) => format!("{} or {}", rest.join(", "), last),
            None => String::new(),
        }
    }
}
