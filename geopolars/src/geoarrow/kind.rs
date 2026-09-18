//! Which geometry a column holds.

/// A GeoArrow native geometry.
///
/// The geometries differ in three facts, all of them `const`, and in nothing
/// else: every one of them is [`Geo`](super::Geo) over a [`Dimension`](super::Dimension).
/// Adding one is a variant here plus an arm in each `match` below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Point,
    LineString,
    Polygon,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Point, Kind::LineString, Kind::Polygon];

    /// The `ARROW:extension:name` this geometry is registered under.
    /// The Python side must spell the same name.
    pub const fn name(self) -> &'static str {
        match self {
            Kind::Point => "geoarrow.point",
            Kind::LineString => "geoarrow.linestring",
            Kind::Polygon => "geoarrow.polygon",
        }
    }

    /// How the geometry is shown in a DataFrame header: `point[xy]`.
    pub const fn display(self) -> &'static str {
        match self {
            Kind::Point => "point",
            Kind::LineString => "linestring",
            Kind::Polygon => "polygon",
        }
    }

    /// How the geometry is spelled as a type: `PointXY`, `LineStringXY`.
    pub const fn type_name(self) -> &'static str {
        match self {
            Kind::Point => "Point",
            Kind::LineString => "LineString",
            Kind::Polygon => "Polygon",
        }
    }

    /// `List` layers between the storage and the coordinate struct.
    ///
    /// This is the whole of how the geometries differ in memory: a point stores
    /// one coordinate per row, a linestring a list of them, and a polygon a list
    /// of those -- one per ring. Expressions walk this rather than asking which
    /// geometry they were handed.
    pub const fn nesting(self) -> u8 {
        match self {
            Kind::Point => 0,
            Kind::LineString => 1,
            Kind::Polygon => 2,
        }
    }

    pub fn from_name(name: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|kind| kind.name() == name)
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
