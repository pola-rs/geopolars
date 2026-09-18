//! Expressions over GeoArrow geometry columns.
//!
//! Polars finds the expression functions by doing a `dlsym` for `_polars_plugin_*`
//! to get symbols out of this compiled library, so how they are split across
//! modules here is for readers only -- nothing has to be registered.
//!
//! Nothing in here names a concrete geometry. An expression asks
//! [`geoarrow::describe`](crate::geoarrow::describe) what it was handed and then
//! either does not care (see [`coords::map_coords`]) or matches on the
//! [`Kind`](crate::geoarrow::Kind) exhaustively.

pub mod affine;
pub mod construct;
pub mod coords;
