//! Views over the Polars storage of GeoArrow geometries,
//! and the arrays kernels build their output in.
//!
//! See the README.

mod coords;
mod dimension;
mod geometry_array;
mod geotraits;
mod linestrings;
mod lists;
pub mod output;
mod points;
mod polygons;
mod unchecked;
mod untrimmed;

use polars_arrow::array::Array;
use polars_core::prelude::{polars_err, PolarsResult};

pub use coords::{
    is_close, is_close_abs, is_ring, same_place, same_place_within, same_place_within_abs,
    same_position, AxisArrays, Coords, Tolerance, ValidCoords, MIN_RING_VERTICES,
};
pub use dimension::GeoDimension;
pub use geometry_array::GeometryArray;
pub use geotraits::{Coord, Geometry};
pub use linestrings::{
    LineStringArray, MultiLineStringArray, ScalarLineString, ScalarMultiLineString,
};
pub use lists::{List, ValidList};
pub use points::{MultiPointArray, PointArray, ScalarMultiPoint, ScalarPoint};
pub use polygons::{MultiPolygonArray, PolygonArray, ScalarMultiPolygon, ScalarPolygon};
pub use unchecked::UncheckedStorage;
pub use untrimmed::Untrimmed;

/// Downcast one level of the storage to the array it has to be.
/// The dtype was already checked, so this failing is a bug.
pub fn downcast<'a, T: 'static>(array: &'a dyn Array, what: &str) -> PolarsResult<&'a T> {
    array.as_any().downcast_ref::<T>().ok_or_else(
        || polars_err!(ComputeError: "expected {what} in geometry storage, got: {:?}", array.dtype()),
    )
}
