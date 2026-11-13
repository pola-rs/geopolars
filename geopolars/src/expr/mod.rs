//! Expressions over GeoArrow geometry columns.
//!
//! Note that organisition into submodules is for readers only,
//! as Polars finds the expression functions by doing a `dlsym` for `_polars_plugin_*`
//!
//!
//! This all just dispatches to `geopolars_core`,
//! which picks a kernel in `geopolars_compute`.

pub mod add_holes;
pub mod add_parts;
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
pub mod hausdorff_line_between;
pub mod is_ccw;
pub mod is_empty;
pub mod is_equal;
pub mod is_ring;
pub mod is_simple;
pub mod is_topologically_equal;
pub mod is_valid;
pub mod join_line_strings;
pub mod length;
pub mod line_interpolate_point;
pub mod offset_curve;
pub mod overlay;
pub mod relate;
pub mod remove_repeats_in_succession;
pub mod reverse;
pub mod rings;
pub mod rings_overlap;
pub mod shortest_line_between;
pub mod simplify;
pub mod snap_to_grid;
pub mod sort_parts;
pub mod to_ccw;
pub mod wkb;
pub mod wkt;
