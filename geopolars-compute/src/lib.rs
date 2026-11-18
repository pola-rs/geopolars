//! Kernels on GeoArrow, taking the views of `geopolars-views`:
//! [native] ones of our own, and [geos], [rsgeo] and [proj] ones forwarding to those libraries.
//! [geoarrow_validity] and [ogc_validity] are the checks of `is_valid_geoarrow` and `is_valid_ogc`.
//! [inner_nulls] moves null values below the rows out to the rows,
//! or marks the parts holding one in a mask instead.
//!
//! See the README.

// Declared first, so that its macros are in scope for every other module.
#[macro_use]
mod const_generic_dispatch;

mod early_coord_check;
mod empty_polygon_exterior;
mod error;
#[cfg(feature = "native")]
pub mod geoarrow_validity;
#[cfg(feature = "geos")]
pub mod geos;
pub mod inner_nulls;
mod metric;
#[cfg(feature = "native")]
pub mod native;
mod new_ring_orientation;
mod null_parts;
#[cfg(all(feature = "native", feature = "geos"))]
pub mod ogc_validity;
mod on_row_error;
mod passing;
#[cfg(any(feature = "native", feature = "rsgeo", feature = "geos"))]
// Some helpers are only for `native`, others only for `geos` and `rsgeo`.
#[cfg_attr(
    not(all(feature = "native", any(feature = "geos", feature = "rsgeo"))),
    allow(dead_code)
)]
mod per_row;
pub mod proj;
#[cfg(feature = "rsgeo")]
pub mod rsgeo;
mod short_lines;
pub mod utils;

pub use early_coord_check::EarlyCoordCheck;
pub use empty_polygon_exterior::EmptyPolygonExterior;
#[cfg(any(feature = "native", feature = "geos", feature = "rsgeo"))]
use error::row_or_null;
pub use error::{GeoComputeError, GeoComputeResult};
pub use metric::{BaseMetric, GeodesicMetric, Planar, WithHeight};
pub use new_ring_orientation::{NewHoles, NewRingOrientation};
pub use null_parts::NullParts;
pub use on_row_error::OnRowError;
pub use short_lines::ShortLines;
