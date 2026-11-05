//! Kernels of our own, working straight off the GeoArrow buffers.

mod affine;
mod area;
mod bounds;
mod centroid;
mod coordinate_selection;
mod count_coordinates;
pub mod distance;
mod fix_out_of_bounds;
mod has_repeats_in_succession;
mod is_ccw;
mod is_close;
mod is_empty;
mod is_ring;
mod is_simple;
// For `geoarrow_validity` and `ogc_validity`.
mod is_valid;
mod join_line_strings;
mod length;
mod remove_repeats_in_succession;
mod reverse;
mod rings;
mod selected_coords;
mod snap_to_grid;
mod sort_parts;
mod too_short;

pub use affine::{
    affine_line_strings, affine_multi_line_strings, affine_multi_points, affine_multi_polygons,
    affine_points, affine_polygons, AffineMatrix, RowShift,
};
pub use area::{multi_polygon_areas, polygon_areas, AreaMetric};
pub use bounds::{
    box_array, line_string_bounds, line_string_column_extent, multi_line_string_bounds,
    multi_line_string_column_extent, multi_line_string_part_bounds, multi_point_bounds,
    multi_point_column_extent, multi_point_part_bounds, multi_polygon_bounds,
    multi_polygon_column_extent, multi_polygon_part_bounds, point_bounds, point_column_extent,
    polygon_bounds, polygon_column_extent, BoxOptions, Extent,
};
pub use centroid::{
    line_string_centroids, multi_line_string_centroids, multi_polygon_centroids, polygon_centroids,
};
pub(crate) use coordinate_selection::which_are_not_selected;
pub use coordinate_selection::{selected_coords, Bounds, CoordinateSelection, NonFinite};
pub use count_coordinates::{
    line_string_coordinate_counts, multi_line_string_coordinate_counts,
    multi_point_coordinate_counts, multi_polygon_coordinate_counts, point_coordinate_counts,
    polygon_coordinate_counts,
};
pub use distance::{against_one, pairwise, DistanceMetric};
pub use fix_out_of_bounds::{
    line_strings_with_clamped_coords, multi_line_strings_with_clamped_coords,
    multi_points_with_clamped_coords, multi_polygons_with_clamped_coords,
    points_with_clamped_coords, polygons_with_clamped_coords, OutOfBoundsAction,
};
pub use has_repeats_in_succession::{
    line_strings_have_repeats_in_succession, multi_line_strings_have_repeats_in_succession,
    multi_points_have_repeats_in_succession, multi_polygons_have_repeats_in_succession,
    points_have_repeats_in_succession, polygons_have_repeats_in_succession, Coincidence,
};
pub use is_ccw::{
    line_strings_are_ccw, line_strings_to_ccw, multi_line_strings_are_ccw,
    multi_line_strings_to_ccw, multi_polygons_are_ccw, multi_polygons_to_ccw, polygons_are_ccw,
    polygons_to_ccw,
};
pub use is_close::{
    line_strings_are_close, multi_line_strings_are_close, multi_points_are_close,
    multi_polygons_are_close, points_are_close, polygons_are_close,
};
pub use is_empty::{
    line_strings_are_empty, multi_line_strings_are_empty, multi_points_are_empty,
    multi_polygons_are_empty, points_are_empty, polygons_are_empty,
};
pub use is_ring::{
    line_strings_are_rings, multi_line_strings_are_rings, multi_polygons_have_only_rings,
    polygons_have_only_rings,
};
pub use is_simple::{multi_points_are_simple, points_are_simple};
pub(crate) use is_valid::rings_are_closed;
// Only for `ogc_validity`.
#[cfg(feature = "geos")]
pub(crate) use is_valid::{line_shapes_are_valid, polygon_shapes_are_valid};
pub use join_line_strings::{multi_line_strings_joined, multi_line_strings_split_at_gaps, Gaps};
pub use length::{line_string_lengths, multi_line_string_lengths, LengthMetric};
pub use remove_repeats_in_succession::{
    line_strings_without_repeats_in_succession, multi_line_strings_without_repeats_in_succession,
    multi_points_without_repeats_in_succession, multi_polygons_without_repeats_in_succession,
    polygons_without_repeats_in_succession, CollapsedLines, LineOptions,
};
pub use reverse::{
    reverse_line_strings, reverse_multi_line_strings, reverse_multi_points, reverse_multi_polygons,
    reverse_polygons,
};
pub use rings::{
    close_open_rings, multi_polygon_exteriors, multi_polygon_interiors, polygon_exteriors,
    polygon_interiors, polygons_with_added_holes,
};
pub use selected_coords::{
    line_strings_with_selected, multi_line_strings_with_selected, multi_points_with_selected,
    multi_polygons_with_selected, points_with_selected, polygons_with_selected, SelectedAction,
};
pub use snap_to_grid::{
    line_strings_snapped_to_grid, multi_line_strings_snapped_to_grid, multi_points_snapped_to_grid,
    multi_polygons_snapped_to_grid, points_snapped_to_grid, polygons_snapped_to_grid, Grid,
    RoundMode,
};
pub use sort_parts::{
    multi_line_strings_sorted, multi_points_sorted, multi_polygons_sorted,
    polygons_with_holes_sorted, PartSorter,
};
pub use too_short::TooShort;
