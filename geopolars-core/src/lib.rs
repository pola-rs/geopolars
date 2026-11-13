//! Picks the kernel for each kind of geometry.
//!
//! See the README.

mod add_holes;
mod add_parts;
mod affine;
mod area;
mod bounds;
mod buffer;
mod centroid;
mod close_rings;
mod construct;
mod convex_hull;
mod count_coordinates;
mod crs;
mod distance;
mod filter_coordinates;
mod fix_out_of_bounds;
mod has_repeats_in_succession;
mod hausdorff_line_between;
mod is_ccw;
mod is_empty;
mod is_equal;
mod is_ring;
mod is_simple;
mod is_topologically_equal;
mod is_valid;
mod join_line_strings;
mod kinds;
mod length;
mod line_interpolate_point;
mod offset_curve;
mod overlay;
mod parts;
mod relate;
mod remove_repeats_in_succession;
mod reverse;
mod rings;
mod rings_overlap;
mod shortest_line_between;
mod simplify;
mod snap_to_grid;
mod sort_parts;
mod to_ccw;

pub use add_holes::{add_holes, EmptyExterior};
pub use add_parts::{add_parts, EmptyParts};
pub use affine::{affine_transform, affine_transform_about, affine_transform_then_shift};
pub use area::areas;
pub use bounds::{bounds, total_bounds};
pub use buffer::buffers;
pub use centroid::centroids;
pub use close_rings::close_rings;
pub use construct::{assembled_multi_polygons, assembled_polygons, validated_storage};
pub use convex_hull::convex_hulls;
pub use count_coordinates::coordinate_counts;
pub use crs::reproject;
pub use distance::{distances_to_one, pairwise_distances};
pub use filter_coordinates::with_selected_coords;
pub use fix_out_of_bounds::with_clamped_coords;
pub use has_repeats_in_succession::have_repeats_in_succession;
pub use hausdorff_line_between::{hausdorff_lines_to_one, pairwise_hausdorff_lines};
pub use is_ccw::are_ccw;
pub use is_empty::are_empty;
pub use is_equal::are_equal;
pub use is_ring::are_rings;
pub use is_simple::are_simple;
pub use is_topologically_equal::are_topologically_equal;
pub use is_valid::{are_valid, Validity};
pub use join_line_strings::{join_line_strings, JoinGaps};
pub use kinds::{LineKind, NestedKind, PolygonKind, SegmentKind};
pub use length::lengths;
pub use line_interpolate_point::interpolated_points;
pub use offset_curve::offset_curves;
pub use overlay::{overlay, OverlayOptions};
pub use parts::{CloseRings, NullParts};
pub use relate::relate;
pub use remove_repeats_in_succession::without_repeats_in_succession;
pub use reverse::reverse;
pub use rings::{exteriors, interiors};
pub use rings_overlap::{pairwise_rings_overlap, rings_overlap_one};
pub use shortest_line_between::{pairwise_shortest_lines, shortest_lines_to_one};
pub use simplify::simplify;
pub use snap_to_grid::snap_to_grid;
pub use sort_parts::sort_parts;
pub use to_ccw::to_ccw;

use geopolars_compute::GeoComputeResult;
use polars_core::prelude::{polars_err, PolarsResult};

/// Turns the result of a `geopolars_compute` kernel into a [`PolarsResult`].
/// The orphan rule rules out a `From` impl here,
/// and `geopolars_compute` should not depend on `polars_err` to provide one itself.
trait IntoPolarsResult<T> {
    fn into_polars(self) -> PolarsResult<T>;
}

impl<T> IntoPolarsResult<T> for GeoComputeResult<T> {
    fn into_polars(self) -> PolarsResult<T> {
        self.map_err(|e| polars_err!(ComputeError: "{e}"))
    }
}
