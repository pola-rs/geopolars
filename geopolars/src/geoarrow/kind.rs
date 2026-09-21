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
    MultiPoint,
}

impl Kind {
    pub const ALL: [Kind; 4] = [
        Kind::Point,
        Kind::LineString,
        Kind::Polygon,
        Kind::MultiPoint,
    ];

    /// The `ARROW:extension:name` this geometry is registered under.
    /// The Python side must spell the same name.
    pub const fn name(self) -> &'static str {
        match self {
            Kind::Point => "geoarrow.point",
            Kind::LineString => "geoarrow.linestring",
            Kind::Polygon => "geoarrow.polygon",
            Kind::MultiPoint => "geoarrow.multipoint",
        }
    }

    /// How the geometry is shown in a DataFrame header: `point[xy]`.
    pub const fn display(self) -> &'static str {
        match self {
            Kind::Point => "point",
            Kind::LineString => "linestring",
            Kind::Polygon => "polygon",
            Kind::MultiPoint => "multipoint",
        }
    }

    /// How the geometry is spelled as a type: `PointXY`, `LineStringXY`.
    pub const fn type_name(self) -> &'static str {
        match self {
            Kind::Point => "Point",
            Kind::LineString => "LineString",
            Kind::Polygon => "Polygon",
            Kind::MultiPoint => "MultiPoint",
        }
    }

    /// Tells us how deeply nested the data type is.
    /// A geoarrow.linestring is a collection of points, so it is 1.
    /// A geoarrow.polygon is a collection of linestings, so it is 2.
    ///
    /// Warning: This does not uniquely identify a geometry kind:
    /// E.g. geoarrow.linestring and geoarrow.multipoint both have a nesting of 2.
    /// The extension name is what tells them apart,
    /// which is why [`describe`](super::describe) reads the name and derives the nesting,
    /// rather than the other way round.
    pub const fn nesting(self) -> u8 {
        match self {
            Kind::Point => 0,
            Kind::LineString | Kind::MultiPoint => 1,
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
