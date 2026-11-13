//! The kinds an operation accepts, narrowed down from [`Kind`].
//! An operation that takes some kinds, but not all of them, refuses the rest by narrowing.
//! The plugin narrows with the same call at plan time and before dispatch,
//! so the two cannot disagree on which kinds the operation takes.
//! [`sort_parts`](crate::sort_parts) is the one exception,
//! as it also refuses a linestring at run time unless it rotates rings.

use geopolars_views::Kind;
use polars_core::prelude::{polars_err, PolarsResult};

/// A linestring or multilinestring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    LineString,
    MultiLineString,
}

impl LineKind {
    const ALL: [Self; 2] = [Self::LineString, Self::MultiLineString];

    /// Narrows `kind`, or refuses any other kind in an error naming `operation`.
    pub fn new(operation: &str, kind: Kind) -> PolarsResult<Self> {
        narrowed(operation, kind, Self::ALL, Self::kind)
    }

    pub const fn kind(self) -> Kind {
        match self {
            Self::LineString => Kind::LineString,
            Self::MultiLineString => Kind::MultiLineString,
        }
    }
}

/// A polygon or multipolygon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolygonKind {
    Polygon,
    MultiPolygon,
}

impl PolygonKind {
    const ALL: [Self; 2] = [Self::Polygon, Self::MultiPolygon];

    /// Narrows `kind`, or refuses any other kind in an error naming `operation`.
    pub fn new(operation: &str, kind: Kind) -> PolarsResult<Self> {
        narrowed(operation, kind, Self::ALL, Self::kind)
    }

    pub const fn kind(self) -> Kind {
        match self {
            Self::Polygon => Kind::Polygon,
            Self::MultiPolygon => Kind::MultiPolygon,
        }
    }
}

/// A geometry made of segments: a (multi)linestring or (multi)polygon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    LineString,
    Polygon,
    MultiLineString,
    MultiPolygon,
}

impl SegmentKind {
    const ALL: [Self; 4] = [
        Self::LineString,
        Self::Polygon,
        Self::MultiLineString,
        Self::MultiPolygon,
    ];

    /// Narrows `kind`, or refuses any other kind in an error naming `operation`.
    pub fn new(operation: &str, kind: Kind) -> PolarsResult<Self> {
        narrowed(operation, kind, Self::ALL, Self::kind)
    }

    pub const fn kind(self) -> Kind {
        match self {
            Self::LineString => Kind::LineString,
            Self::Polygon => Kind::Polygon,
            Self::MultiLineString => Kind::MultiLineString,
            Self::MultiPolygon => Kind::MultiPolygon,
        }
    }
}

/// Any geometry but a point, so one whose storage is a list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NestedKind {
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
}

impl NestedKind {
    const ALL: [Self; 5] = [
        Self::LineString,
        Self::Polygon,
        Self::MultiPoint,
        Self::MultiLineString,
        Self::MultiPolygon,
    ];

    /// Narrows `kind`, or refuses any other kind in an error naming `operation`.
    pub fn new(operation: &str, kind: Kind) -> PolarsResult<Self> {
        narrowed(operation, kind, Self::ALL, Self::kind)
    }

    pub const fn kind(self) -> Kind {
        match self {
            Self::LineString => Kind::LineString,
            Self::Polygon => Kind::Polygon,
            Self::MultiPoint => Kind::MultiPoint,
            Self::MultiLineString => Kind::MultiLineString,
            Self::MultiPolygon => Kind::MultiPolygon,
        }
    }
}

fn narrowed<T: Copy, const N: usize>(
    operation: &str,
    kind: Kind,
    all: [T; N],
    kind_of: fn(T) -> Kind,
) -> PolarsResult<T> {
    all.into_iter()
        .find(|&narrowed| kind_of(narrowed) == kind)
        .ok_or_else(|| {
            polars_err!(
                SchemaMismatch: "{operation} expects a {} column, got: `{}`",
                Kind::names_of(all.map(kind_of)), kind.name()
            )
        })
}
