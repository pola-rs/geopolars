//! Which coordinates a geometry carries.

use polars_core::prelude::{DataType, Field};

/// Which coordinate dimensions a geometry carries.
///
/// Because the spec fixes the order of the coordinate data,
/// this also determines its dimension.
///
/// Note this is one enum rather than four separate extension types.
/// An enum is cheaper to match on than casting to types.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    // TODO: Rename to GeoDimension, just Dimension trips up rust-analyzer
    XY,
    XYZ,
    XYM,
    XYZM,
}

/// Const impl
impl Dimension {
    pub const ALL: [Dimension; 4] = [
        Dimension::XY,
        Dimension::XYZ,
        Dimension::XYM,
        Dimension::XYZM,
    ];

    /// The storage struct's field names, in the order the spec requires.
    pub const fn field_names(self) -> &'static [&'static str] {
        match self {
            Dimension::XY => &["x", "y"],
            Dimension::XYZ => &["x", "y", "z"],
            Dimension::XYM => &["x", "y", "m"],
            Dimension::XYZM => &["x", "y", "z", "m"],
        }
    }

    /// How the dimension is spelled in a type name: `point[xy]`, `LineStringXY`.
    /// The same thing [`Dimension::field_names`] says, but as one `const` string.
    pub const fn tag(self) -> &'static str {
        match self {
            Dimension::XY => "xy",
            Dimension::XYZ => "xyz",
            Dimension::XYM => "xym",
            Dimension::XYZM => "xyzm",
        }
    }

    /// `true` if this dimension has a `z` (elevation) coordinate.
    pub const fn has_z(self) -> bool {
        matches!(self, Dimension::XYZ | Dimension::XYZM)
    }

    /// `true` if this dimension has an `m` (measure) value.
    #[allow(dead_code)]
    pub const fn has_m(self) -> bool {
        matches!(self, Dimension::XYM | Dimension::XYZM)
    }

    /// The separated-coordinate struct an array of these coordinates is stored as.
    pub fn coordinates(self) -> DataType {
        DataType::Struct(
            self.field_names()
                .iter()
                .map(|name| Field::new((*name).into(), DataType::Float64))
                .collect(),
        )
    }

    /// The storage of a geometry that nests its coordinates `nesting` `List`
    /// layers deep. See [`Kind::nesting`](super::Kind::nesting).
    pub fn storage(self, nesting: u8) -> DataType {
        (0..nesting).fold(self.coordinates(), |inner, _| {
            DataType::List(Box::new(inner))
        })
    }
}

#[cfg(test)]
mod tests {
    //! Only what the Python suite cannot reach.

    use super::*;

    /// Every type name is built from [`Dimension::tag`], which is hand-written,
    /// so check it still agrees with the field list it claims to describe.
    ///
    /// Python cannot: it spells its own names out per class.
    #[test]
    fn the_tag_follows_from_the_field_names() {
        for dim in Dimension::ALL {
            assert_eq!(dim.tag(), dim.field_names().concat());
        }
    }
}
