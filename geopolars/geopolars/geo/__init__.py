"""Provides the functional API for Expressions over GeoArrow geometry columns.
Everything in the 'geo' namespace forwards to here.
"""

from geopolars.geo.affine import (
    affine_transform,
    rotate,
    rotate_about_centroid,
    scale,
    scale_about_centroid,
    skew,
    skew_about_centroid,
    translate,
)
from geopolars.geo.area import area, area_geodesic, area_planar
from geopolars.geo.boundary import boundary
from geopolars.geo.bounds import bounds, total_bounds
from geopolars.geo.buffer import buffer
from geopolars.geo.centroid import centroid
from geopolars.geo.close_rings import CloseRings, close_rings
from geopolars.geo.construct.box import box
from geopolars.geo.construct.linestring import (
    line_string,
    line_string_from_columns,
    line_string_from_vertices,
    multi_line_string,
    multi_line_string_from_columns,
    multi_line_string_from_line_strings,
)
from geopolars.geo.construct.point import (
    multi_point,
    multi_point_from_columns,
    multi_point_from_points,
    point,
)
from geopolars.geo.construct.polygon import (
    multi_polygons_from_coordinates,
    multi_polygons_from_nested_coordinates,
    multi_polygons_from_polygons,
    polygons_from_coordinates,
    polygons_from_nested_coordinates,
    polygons_from_rings_in_separate_columns,
    polygons_from_rings_in_single_column,
)
from geopolars.geo.convex_hull import convex_hull
from geopolars.geo.coordinates import m, x, y, z
from geopolars.geo.count_coordinates import count_coordinates
from geopolars.geo.crs import is_geographic, set_crs, to_crs
from geopolars.geo.distance import distance
from geopolars.geo.exterior import exterior
from geopolars.geo.filter_coordinates import NonFinite, filter_coordinates
from geopolars.geo.fix_out_of_bounds import OutOfBounds, fix_out_of_bounds
from geopolars.geo.force_dimension import force_2d, force_3d, force_m, without_m
from geopolars.geo.has_repeats_in_succession import has_repeats_in_succession
from geopolars.geo.interior import interior
from geopolars.geo.is_ccw import is_ccw
from geopolars.geo.is_close import is_close
from geopolars.geo.is_empty import is_empty
from geopolars.geo.is_ring import is_ring
from geopolars.geo.is_simple import is_simple
from geopolars.geo.is_valid import (
    is_valid_geoarrow,
    is_valid_ogc,
    null_out_invalid_geoarrow,
    null_out_invalid_ogc,
)
from geopolars.geo.length import length
from geopolars.geo.mean_coordinate import mean_coordinate
from geopolars.geo.offset_curve import offset_curve
from geopolars.geo.parts import parts
from geopolars.geo.remove_repeats_in_succession import remove_repeats_in_succession
from geopolars.geo.reverse import reverse
from geopolars.geo.rings_overlap import rings_overlap
from geopolars.geo.simplify import simplify
from geopolars.geo.snap_to_grid import snap_to_grid
from geopolars.geo.to_ccw import to_ccw, to_cw
from geopolars.geo.to_polygon import to_polygon
from geopolars.geo.wkb import from_wkb, to_wkb
from geopolars.geo.wkt import from_wkt, to_wkt
from geopolars.geo.wrap_longitude import wrap_longitude

__all__ = [
    "CloseRings",
    "NonFinite",
    "OutOfBounds",
    "affine_transform",
    "area",
    "area_geodesic",
    "area_planar",
    "boundary",
    "bounds",
    "box",
    "buffer",
    "centroid",
    "close_rings",
    "convex_hull",
    "count_coordinates",
    "distance",
    "exterior",
    "filter_coordinates",
    "fix_out_of_bounds",
    "force_2d",
    "force_3d",
    "force_m",
    "from_wkb",
    "from_wkt",
    "has_repeats_in_succession",
    "interior",
    "is_ccw",
    "is_close",
    "is_empty",
    "is_geographic",
    "is_ring",
    "is_simple",
    "is_valid_geoarrow",
    "is_valid_ogc",
    "length",
    "line_string",
    "line_string_from_columns",
    "line_string_from_vertices",
    "m",
    "mean_coordinate",
    "multi_line_string",
    "multi_line_string_from_columns",
    "multi_line_string_from_line_strings",
    "multi_point",
    "multi_point_from_columns",
    "multi_point_from_points",
    "multi_polygons_from_coordinates",
    "multi_polygons_from_nested_coordinates",
    "multi_polygons_from_polygons",
    "null_out_invalid_geoarrow",
    "null_out_invalid_ogc",
    "offset_curve",
    "parts",
    "point",
    "polygons_from_coordinates",
    "polygons_from_nested_coordinates",
    "polygons_from_rings_in_separate_columns",
    "polygons_from_rings_in_single_column",
    "remove_repeats_in_succession",
    "reverse",
    "rings_overlap",
    "rotate",
    "rotate_about_centroid",
    "scale",
    "scale_about_centroid",
    "set_crs",
    "simplify",
    "skew",
    "skew_about_centroid",
    "snap_to_grid",
    "to_ccw",
    "to_crs",
    "to_cw",
    "to_polygon",
    "to_wkb",
    "to_wkt",
    "total_bounds",
    "translate",
    "without_m",
    "wrap_longitude",
    "x",
    "y",
    "z",
]
