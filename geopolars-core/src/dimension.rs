//! In GeoArrow, a coordinate can be 2D or 3D,
//! and optionally carry a 'measurement'.
//!
//! `Geometry` can be built up from any of these.
//! Dimensional information is carried by the coordinates, and forwarded to the geometries.
//!
//! All variations are defined in [GeoDimension].
//! Note this is one enum rather than four separate rust types.
//! In polars we already work with an abstraction over the type system,
//! and this extends that framework.

use geo_traits::Dimensions;
use polars_core::prelude::{DataType, Field};

/// Which coordinate dimensions a coordinate/geometry carries.
///
/// Because the spec fixes the order of the coordinate data,
/// this also determines its dimension.
///
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeoDimension {
    // This is named 'GeoDimension' and not 'Dimension' because for some reason
    // rust-analyzer really messes up by confusing it with [polars::datatype::Dimension].
    XY,
    XYZ,
    XYM,
    XYZM,
}

/// Const impl
impl GeoDimension {
    pub const ALL: [GeoDimension; 4] = [
        GeoDimension::XY,
        GeoDimension::XYZ,
        GeoDimension::XYM,
        GeoDimension::XYZM,
    ];

    /// The storage struct's field names, in the order the spec requires.
    pub const fn field_names(self) -> &'static [&'static str] {
        match self {
            GeoDimension::XY => &["x", "y"],
            GeoDimension::XYZ => &["x", "y", "z"],
            GeoDimension::XYM => &["x", "y", "m"],
            GeoDimension::XYZM => &["x", "y", "z", "m"],
        }
    }

    /// How the dimension is spelled in a type name: `point[xy]`, `LineStringXY`.
    /// The same thing [`Dimension::field_names`] says, but as one `const` string.
    pub const fn tag(self) -> &'static str {
        match self {
            GeoDimension::XY => "xy",
            GeoDimension::XYZ => "xyz",
            GeoDimension::XYM => "xym",
            GeoDimension::XYZM => "xyzm",
        }
    }

    /// The inverse of [`GeoDimension::tag`].
    pub fn from_tag(tag: &str) -> Option<GeoDimension> {
        GeoDimension::ALL.into_iter().find(|dim| dim.tag() == tag)
    }

    /// `true` if this dimension has a `z` (elevation) coordinate.
    pub const fn has_z(self) -> bool {
        matches!(self, GeoDimension::XYZ | GeoDimension::XYZM)
    }

    /// `true` if this dimension has an `m` (measure) value.
    pub const fn has_m(self) -> bool {
        matches!(self, GeoDimension::XYM | GeoDimension::XYZM)
    }

    /// Finds the dimensions that both have in common.
    /// For example, a PointXYZ and a PointXYM only have XY in common.
    pub const fn shared_with(self, other: GeoDimension) -> GeoDimension {
        match (self.has_z() && other.has_z(), self.has_m() && other.has_m()) {
            (false, false) => GeoDimension::XY,
            (true, false) => GeoDimension::XYZ,
            (false, true) => GeoDimension::XYM,
            (true, true) => GeoDimension::XYZM,
        }
    }

    /// A `geoarrow.box`'s field names, in the order the spec requires:
    /// every minimum, then every maximum.
    pub const fn box_field_names(self) -> &'static [&'static str] {
        match self {
            GeoDimension::XY => &["xmin", "ymin", "xmax", "ymax"],
            GeoDimension::XYZ => &["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"],
            GeoDimension::XYM => &["xmin", "ymin", "mmin", "xmax", "ymax", "mmax"],
            GeoDimension::XYZM => &[
                "xmin", "ymin", "zmin", "mmin", "xmax", "ymax", "zmax", "mmax",
            ],
        }
    }
}

/// The polars dtypes of coordinates, geometries and boxes.
impl GeoDimension {
    /// The separated-coordinate struct an array of these coordinates is stored as.
    pub fn coordinates(self) -> DataType {
        DataType::Struct(
            self.field_names()
                .iter()
                .map(|name| Field::new((*name).into(), DataType::Float64))
                .collect(),
        )
    }

    /// The storage of a geometry that nests its coordinates `nesting` `List` layers deep.
    pub fn storage(self, nesting: u8) -> DataType {
        (0..nesting).fold(self.coordinates(), |inner, _| {
            DataType::List(Box::new(inner))
        })
    }

    /// The struct a `geoarrow.box` of this dimension is stored as.
    pub fn box_storage(self) -> DataType {
        DataType::Struct(
            self.box_field_names()
                .iter()
                .map(|name| Field::new((*name).into(), DataType::Float64))
                .collect(),
        )
    }
}

impl From<GeoDimension> for Dimensions {
    fn from(dim: GeoDimension) -> Self {
        match dim {
            GeoDimension::XY => Dimensions::Xy,
            GeoDimension::XYZ => Dimensions::Xyz,
            GeoDimension::XYM => Dimensions::Xym,
            GeoDimension::XYZM => Dimensions::Xyzm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Consistency check
    #[test]
    fn the_tag_follows_from_the_field_names() {
        for dim in GeoDimension::ALL {
            assert_eq!(dim.tag(), dim.field_names().concat());
        }
    }
}
