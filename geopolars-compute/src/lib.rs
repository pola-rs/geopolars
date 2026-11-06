//! Kernels on GeoArrow, taking the views of `geopolars-core`:
//! [native] ones of our own, and [geos], [rsgeo] and [proj] ones forwarding to those libraries.
//! [geoarrow_validity] and [ogc_validity] are the checks of `is_valid_geoarrow` and `is_valid_ogc`.
//! [inner_nulls] moves null values below the rows out to the rows,
//! or marks the parts holding one in a mask instead.
//!
//! See the README.

/// Dispatches runtime `bool` flags into generic implementations,
/// so that the hot path never branches on them.
/// `with_const_flags!(A = a, B = b; f::<A, B>(args))` calls `f::<A, B>(args)`,
/// with each runtime flag turned into a const `bool` of the name it is bound to.
#[macro_export]
macro_rules! with_const_flags {
    (@ $call:expr;) => {
        $call
    };
    (@ $call:expr; $name:ident = $flag:expr $(, $rest_name:ident = $rest_flag:expr)*) => {
        if $flag {
            const $name: bool = true;
            $crate::with_const_flags!(@ $call; $($rest_name = $rest_flag),*)
        } else {
            const $name: bool = false;
            $crate::with_const_flags!(@ $call; $($rest_name = $rest_flag),*)
        }
    };
    ($($name:ident = $flag:expr),+; $call:expr) => {
        $crate::with_const_flags!(@ $call; $($name = $flag),+)
    };
}

mod error;
#[cfg(feature = "native")]
pub mod geoarrow_validity;
#[cfg(feature = "geos")]
pub mod geos;
pub mod inner_nulls;
mod metric;
#[cfg(feature = "native")]
pub mod native;
#[cfg(all(feature = "native", feature = "geos"))]
pub mod ogc_validity;
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

#[cfg(any(feature = "geos", feature = "rsgeo"))]
use error::row_or_null;
pub use error::{GeoComputeError, GeoComputeResult};
pub use metric::{BaseMetric, GeodesicMetric, Planar, WithHeight};
