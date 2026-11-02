//! Expressions over GeoArrow geometry columns.
//!
//! Polars finds the expression functions by doing a `dlsym` for `_polars_plugin_*`
//! to get symbols out of this compiled library, so how they are split across
//! modules here is for readers only -- nothing has to be registered.
//!
//! Nothing in here names a concrete geometry. An expression asks
//! [`geoarrow::describe`](crate::geoarrow::describe) what it was handed and then
//! matches on the [`Kind`](crate::geoarrow::Kind) exhaustively.

pub mod add_holes;
pub mod affine;
pub mod area;
pub mod bounds;
pub mod buffer;
pub mod centroid;
pub mod close_rings;
pub mod construct;
pub mod convex_hull;
pub mod coords;
pub mod count_coordinates;
pub mod crs;
pub mod distance;
pub mod filter_coordinates;
pub mod fix_out_of_bounds;
pub mod has_repeats_in_succession;
pub mod is_ccw;
pub mod is_close;
pub mod is_empty;
pub mod is_ring;
pub mod is_simple;
pub mod is_valid;
pub mod length;
pub mod offset_curve;
pub mod remove_repeats_in_succession;
pub mod reverse;
pub mod rings;
pub mod rings_overlap;
pub mod simplify;
pub mod snap_to_grid;
pub mod to_ccw;
pub mod wkb;
pub mod wkt;

use geopolars_compute::GeoComputeResult;
use polars::prelude::{polars_err, PolarsResult};

/// Turns the result of a `geopolars_compute` kernel into a [`PolarsResult`].
/// Can't just do Into<> because orphan rule,
/// and don't want to bring polars_err into scope of geopolars_compute.
trait IntoPolarsResult<T> {
    fn into_polars(self) -> PolarsResult<T>;
}

impl<T> IntoPolarsResult<T> for GeoComputeResult<T> {
    fn into_polars(self) -> PolarsResult<T> {
        self.map_err(|e| polars_err!(ComputeError: "{e}"))
    }
}
