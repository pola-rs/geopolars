//! Picks the kernel for each kind of geometry.
//!
//! See the README.

mod add_holes;
mod add_parts;
mod affine;
mod area;
mod bounds;
mod centroid;
mod close_rings;
mod convex_hull;
mod count_coordinates;
mod parts;

pub use add_holes::{add_holes, EmptyExterior};
pub use add_parts::{add_parts, EmptyParts};
pub use affine::affine_transform;
pub use area::areas;
pub use bounds::{bounds, column_extent};
pub use centroid::centroids;
pub use close_rings::close_rings;
pub use convex_hull::convex_hulls;
pub use count_coordinates::coordinate_counts;
pub use parts::{
    closed_rings, lines_per_row, one_ring_per_row, rows_validity, validated_polygons, CloseRings,
    NullParts,
};

use geopolars_compute::GeoComputeResult;
use geopolars_views::Kind;
use polars_core::prelude::{polars_err, PolarsError, PolarsResult};

/// For a kind the plugin already refused at plan time.
fn refused(operation: &str, kind: Kind) -> PolarsError {
    polars_err!(SchemaMismatch: "{operation} does not accept a `{}` column", kind.name())
}

/// Turns the result of a `geopolars_compute` kernel into a [`PolarsResult`].
/// Can't just do Into<> because orphan rule.
trait IntoPolarsResult<T> {
    fn into_polars(self) -> PolarsResult<T>;
}

impl<T> IntoPolarsResult<T> for GeoComputeResult<T> {
    fn into_polars(self) -> PolarsResult<T> {
        self.map_err(|e| polars_err!(ComputeError: "{e}"))
    }
}
