"""Provides `geo` expression namespace API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

from geopolars.geo import (
    affine,
    coordinates,
    force_dimension,
    wkb,
    wkt,
)
from geopolars.geo.construct import linestring, point, polygon

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType
    from geopolars.geo.close_rings import CloseRings
    from geopolars.geo.filter_coordinates import NonFiniteAction
    from geopolars.geo.fix_out_of_bounds import (
        CrsBounds,
        OutOfBounds,
        OutOfBoundsAction,
    )
    from geopolars.geo.intersection import Keep
    from geopolars.geo.snap_to_grid import RoundMode


@pl.api.register_expr_namespace("geo")
class ExprGeoNameSpace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def line_string(self, *, crs: str | None = None) -> pl.Expr:
        return linestring.line_string_from_vertices(self._expr, crs=crs)

    def multi_point(self, *, crs: str | None = None) -> pl.Expr:
        return point.multi_point_from_points(self._expr, crs=crs)

    def multi_line_string(self, *, crs: str | None = None) -> pl.Expr:
        return linestring.multi_line_string_from_line_strings(self._expr, crs=crs)

    def polygons_from_rings_in_single_column(
        self, *, crs: str | None = None, close: CloseRings | None = None
    ) -> pl.Expr:
        return polygon.polygons_from_rings_in_single_column(
            self._expr, crs=crs, close=close
        )

    def polygons_from_rings_in_separate_columns(
        self,
        holes: IntoExprColumn | None = None,
        *,
        crs: str | None = None,
        close: CloseRings | None = None,
        null_parts: Literal["drop", "null"] = "null",
    ) -> pl.Expr:
        return polygon.polygons_from_rings_in_separate_columns(
            self._expr, holes, crs=crs, close=close, null_parts=null_parts
        )

    def polygons_from_coordinates(
        self,
        y: IntoExprColumn,
        z: IntoExprColumn | None = None,
        m: IntoExprColumn | None = None,
        *,
        crs: str | None = None,
        close: CloseRings | None = None,
    ) -> pl.Expr:
        return polygon.polygons_from_coordinates(
            self._expr, y, z, m, crs=crs, close=close
        )

    def polygons_from_nested_coordinates(
        self,
        y: IntoExprColumn,
        z: IntoExprColumn | None = None,
        m: IntoExprColumn | None = None,
        *,
        crs: str | None = None,
        close: CloseRings | None = None,
    ) -> pl.Expr:
        return polygon.polygons_from_nested_coordinates(
            self._expr, y, z, m, crs=crs, close=close
        )

    def multi_polygons_from_polygons(
        self, *, crs: str | None = None, close: CloseRings | None = None
    ) -> pl.Expr:
        return polygon.multi_polygons_from_polygons(self._expr, crs=crs, close=close)

    def multi_polygons_from_coordinates(
        self,
        y: IntoExprColumn,
        z: IntoExprColumn | None = None,
        m: IntoExprColumn | None = None,
        *,
        crs: str | None = None,
        close: CloseRings | None = None,
    ) -> pl.Expr:
        return polygon.multi_polygons_from_coordinates(
            self._expr, y, z, m, crs=crs, close=close
        )

    def multi_polygons_from_nested_coordinates(
        self,
        y: IntoExprColumn,
        z: IntoExprColumn | None = None,
        m: IntoExprColumn | None = None,
        *,
        crs: str | None = None,
        close: CloseRings | None = None,
    ) -> pl.Expr:
        return polygon.multi_polygons_from_nested_coordinates(
            self._expr, y, z, m, crs=crs, close=close
        )

    def translate(self, offset: affine.Offset | IntoExprColumn) -> pl.Expr:
        return affine.translate(self._expr, offset)

    def affine_transform(
        self,
        matrix: Sequence[float],
        origin: tuple[float, float] | tuple[float, float, float] | IntoExprColumn = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> pl.Expr:
        return affine.affine_transform(self._expr, matrix, origin=origin)

    def rotate(
        self,
        amount: float,
        unit: Literal["deg", "pi"] = "deg",
        axis: Literal["x", "y", "z"] = "z",
        origin: tuple[float, float] | tuple[float, float, float] | IntoExprColumn = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> pl.Expr:
        return affine.rotate(self._expr, amount, unit=unit, axis=axis, origin=origin)

    def rotate_about_centroid(
        self,
        amount: float,
        unit: Literal["deg", "pi"] = "deg",
        axis: Literal["x", "y", "z"] = "z",
    ) -> pl.Expr:
        return affine.rotate_about_centroid(self._expr, amount, unit=unit, axis=axis)

    def skew(
        self,
        xs: float = 0.0,
        ys: float = 0.0,
        unit: Literal["deg", "pi"] = "deg",
        origin: tuple[float, float] | tuple[float, float, float] | IntoExprColumn = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> pl.Expr:
        return affine.skew(self._expr, xs=xs, ys=ys, unit=unit, origin=origin)

    def skew_about_centroid(
        self, xs: float = 0.0, ys: float = 0.0, unit: Literal["deg", "pi"] = "deg"
    ) -> pl.Expr:
        return affine.skew_about_centroid(self._expr, xs=xs, ys=ys, unit=unit)

    def scale(
        self,
        xfact: float = 1.0,
        yfact: float = 1.0,
        zfact: float = 1.0,
        origin: tuple[float, float] | tuple[float, float, float] | IntoExprColumn = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> pl.Expr:
        return affine.scale(
            self._expr, xfact=xfact, yfact=yfact, zfact=zfact, origin=origin
        )

    def scale_about_centroid(
        self, xfact: float = 1.0, yfact: float = 1.0, zfact: float = 1.0
    ) -> pl.Expr:
        return affine.scale_about_centroid(
            self._expr, xfact=xfact, yfact=yfact, zfact=zfact
        )

    def snap_to_grid(
        self,
        spacing: tuple[float, float] | tuple[float, float, float],
        origin: tuple[float, float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        mode: RoundMode = "half_to_even",
    ) -> pl.Expr:
        from geopolars.geo.snap_to_grid import snap_to_grid

        return snap_to_grid(self._expr, spacing, origin=origin, mode=mode)

    def set_crs(self, crs: str, *, force: bool = False) -> pl.Expr:
        from geopolars.geo.crs import set_crs

        return set_crs(self._expr, crs, force=force)

    def to_crs(
        self,
        crs: str,
        *,
        allow_projection_with_unscaled_z: bool = False,
        assume_z_in_target_crs: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.crs import to_crs

        return to_crs(
            self._expr,
            crs,
            allow_projection_with_unscaled_z=allow_projection_with_unscaled_z,
            assume_z_in_target_crs=assume_z_in_target_crs,
        )

    def is_geographic(self, *, ignore_errors: bool = False) -> pl.Expr:
        from geopolars.geo.crs import is_geographic

        return is_geographic(self._expr, ignore_errors=ignore_errors)

    def mean_coordinate(self) -> pl.Expr:
        from geopolars.geo.mean_coordinate import mean_coordinate

        return mean_coordinate(self._expr)

    def centroid(self) -> pl.Expr:
        from geopolars.geo.centroid import centroid

        return centroid(self._expr)

    def exterior(self) -> pl.Expr:
        from geopolars.geo.exterior import exterior

        return exterior(self._expr)

    def interior(self) -> pl.Expr:
        from geopolars.geo.interior import interior

        return interior(self._expr)

    def reverse(self, *, reverse_holes: bool = False, nested: bool = True) -> pl.Expr:
        from geopolars.geo.reverse import reverse

        return reverse(self._expr, reverse_holes=reverse_holes, nested=nested)

    def convex_hull(
        self,
        *,
        ignore_per_row_errors: bool = False,
        ccw_outer: bool = False,
        early_finite_check: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.convex_hull import convex_hull

        return convex_hull(
            self._expr,
            ignore_per_row_errors=ignore_per_row_errors,
            ccw_outer=ccw_outer,
            early_finite_check=early_finite_check,
        )

    def buffer(
        self,
        distance: float | IntoExprColumn,
        *,
        quadrant_segments: int = 8,
        cap_style: Literal["round", "flat", "square"] | None = None,
        join_style: Literal["round", "mitre", "bevel"] = "round",
        mitre_limit: float | None = None,
        side: Literal["both", "left", "right"] = "both",
        allow_splitting: bool = False,
        ccw_outer: bool = False,
        ignore_per_row_errors: bool = False,
        early_finite_check: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.buffer import buffer

        return buffer(
            self._expr,
            distance,
            quadrant_segments=quadrant_segments,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
            side=side,
            allow_splitting=allow_splitting,
            ccw_outer=ccw_outer,
            ignore_per_row_errors=ignore_per_row_errors,
            early_finite_check=early_finite_check,
        )

    def offset_curve(
        self,
        distance: float | IntoExprColumn,
        *,
        quadrant_segments: int = 8,
        join_style: Literal["round", "mitre", "bevel"] = "round",
        mitre_limit: float | None = None,
        allow_splitting: bool = False,
        ignore_per_row_errors: bool = False,
        early_finite_check: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.offset_curve import offset_curve

        return offset_curve(
            self._expr,
            distance,
            quadrant_segments=quadrant_segments,
            join_style=join_style,
            mitre_limit=mitre_limit,
            allow_splitting=allow_splitting,
            ignore_per_row_errors=ignore_per_row_errors,
            early_finite_check=early_finite_check,
        )

    def line_interpolate_point(
        self,
        distance: float | IntoExprColumn,
        *,
        mode: Literal["absolute", "normalized"] = "absolute",
        ignore_per_row_errors: bool = False,
        early_finite_check: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.line_interpolate_point import line_interpolate_point

        return line_interpolate_point(
            self._expr,
            distance,
            mode=mode,
            ignore_per_row_errors=ignore_per_row_errors,
            early_finite_check=early_finite_check,
        )

    def simplify(
        self,
        tolerance: float,
        *,
        preserve_topology: bool = False,
        ignore_per_row_errors: bool = False,
        early_finite_check: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.simplify import simplify

        return simplify(
            self._expr,
            tolerance,
            preserve_topology=preserve_topology,
            ignore_per_row_errors=ignore_per_row_errors,
            early_finite_check=early_finite_check,
        )

    def boundary(self) -> pl.Expr:
        from geopolars.geo.boundary import boundary

        return boundary(self._expr)

    def add_holes(
        self,
        holes: IntoExprColumn,
        *,
        close: CloseRings | None = None,
        null_parts: Literal["drop", "null"] = "null",
        empty_exterior: Literal["null", "keep"] = "null",
    ) -> pl.Expr:
        from geopolars.geo.add_holes import add_holes

        return add_holes(
            self._expr,
            holes,
            close=close,
            null_parts=null_parts,
            empty_exterior=empty_exterior,
        )

    def add_parts(
        self,
        parts: IntoExprColumn,
        *,
        null_parts: Literal["drop", "null"] = "null",
        empty_parts: Literal["keep", "drop"] = "keep",
    ) -> pl.Expr:
        from geopolars.geo.add_parts import add_parts

        return add_parts(
            self._expr,
            parts,
            null_parts=null_parts,
            empty_parts=empty_parts,
        )

    def parts(self) -> pl.Expr:
        from geopolars.geo.parts import parts

        return parts(self._expr)

    def sort_parts(
        self,
        *,
        descending: bool = False,
        rings: Literal["keep", "rotate"] = "keep",
        ignore_per_row_errors: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.sort_parts import sort_parts

        return sort_parts(
            self._expr,
            descending=descending,
            rings=rings,
            ignore_per_row_errors=ignore_per_row_errors,
        )

    def join_line_strings(
        self,
        *,
        gaps: Literal["null", "bridge", "split"] = "null",
        drop_repeated_joins: bool = False,
        abs_tol: float | None = None,
        rel_tol: float | None = None,
        unique_z: bool = True,
        unique_m: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.join_line_strings import join_line_strings

        return join_line_strings(
            self._expr,
            gaps=gaps,
            drop_repeated_joins=drop_repeated_joins,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            unique_z=unique_z,
            unique_m=unique_m,
        )

    def is_ring(self) -> pl.Expr:
        from geopolars.geo.is_ring import is_ring

        return is_ring(self._expr)

    def is_simple(self) -> pl.Expr:
        from geopolars.geo.is_simple import is_simple

        return is_simple(self._expr)

    def rings_overlap(
        self,
        other: IntoExprColumn,
        *,
        include_touching: bool = False,
        ignore_per_row_errors: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.rings_overlap import rings_overlap

        return rings_overlap(
            self._expr,
            other,
            include_touching=include_touching,
            ignore_per_row_errors=ignore_per_row_errors,
        )

    def is_equal(
        self,
        other: IntoExprColumn,
        *,
        abs_tol: float | None = None,
        rel_tol: float | None = None,
        nans_equal: bool = False,
        unique_z: bool = True,
        unique_m: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.is_equal import is_equal

        return is_equal(
            self._expr,
            other,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            nans_equal=nans_equal,
            unique_z=unique_z,
            unique_m=unique_m,
        )

    def is_topologically_equal(
        self,
        other: IntoExprColumn,
        *,
        ignore_per_row_errors: bool = False,
        early_finite_check: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.is_topologically_equal import is_topologically_equal

        return is_topologically_equal(
            self._expr,
            other,
            ignore_per_row_errors=ignore_per_row_errors,
            early_finite_check=early_finite_check,
        )

    def relate(
        self,
        other: IntoExprColumn,
        *,
        ignore_per_row_errors: bool = False,
        early_finite_check: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.relate import relate

        return relate(
            self._expr,
            other,
            ignore_per_row_errors=ignore_per_row_errors,
            early_finite_check=early_finite_check,
        )

    def shortest_line_between(
        self,
        other: IntoExprColumn,
        *,
        in_crs: str | None = None,
        ignore_per_row_errors: bool = False,
        early_finite_check: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.shortest_line_between import shortest_line_between

        return shortest_line_between(
            self._expr,
            other,
            in_crs=in_crs,
            ignore_per_row_errors=ignore_per_row_errors,
            early_finite_check=early_finite_check,
        )

    def intersection(
        self,
        other: IntoExprColumn,
        *,
        keep: Keep | None = None,
        allow_splitting: bool = False,
        ccw_outer: bool = False,
        ignore_per_row_errors: bool = False,
        early_finite_check: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.intersection import intersection

        return intersection(
            self._expr,
            other,
            keep=keep,
            allow_splitting=allow_splitting,
            ccw_outer=ccw_outer,
            ignore_per_row_errors=ignore_per_row_errors,
            early_finite_check=early_finite_check,
        )

    def difference(
        self,
        other: IntoExprColumn,
        *,
        allow_splitting: bool = False,
        ccw_outer: bool = False,
        ignore_per_row_errors: bool = False,
        early_finite_check: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.difference import difference

        return difference(
            self._expr,
            other,
            allow_splitting=allow_splitting,
            ccw_outer=ccw_outer,
            ignore_per_row_errors=ignore_per_row_errors,
            early_finite_check=early_finite_check,
        )

    def hausdorff_line_between(
        self,
        other: IntoExprColumn,
        *,
        non_finite: Literal["raise", "drop", "drop_empty"] = "raise",
        ignore_z: bool = False,
        ignore_m: bool = False,
        ignore_per_row_errors: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.hausdorff_line_between import hausdorff_line_between

        return hausdorff_line_between(
            self._expr,
            other,
            non_finite=non_finite,
            ignore_z=ignore_z,
            ignore_m=ignore_m,
            ignore_per_row_errors=ignore_per_row_errors,
        )

    def is_ccw(self, *, holes_cw: bool = True) -> pl.Expr:
        from geopolars.geo.is_ccw import is_ccw

        return is_ccw(self._expr, holes_cw=holes_cw)

    def to_ccw(self, *, holes_cw: bool = True) -> pl.Expr:
        from geopolars.geo.to_ccw import to_ccw

        return to_ccw(self._expr, holes_cw=holes_cw)

    def to_cw(self, *, holes_ccw: bool = True) -> pl.Expr:
        from geopolars.geo.to_ccw import to_cw

        return to_cw(self._expr, holes_ccw=holes_ccw)

    def close_rings(self, *, snap_z: bool = False, snap_m: bool = False) -> pl.Expr:
        from geopolars.geo.close_rings import close_rings

        return close_rings(self._expr, snap_z=snap_z, snap_m=snap_m)

    def filter_coordinates(
        self,
        *,
        select_non_finite: bool = True,
        select_empty: bool = True,
        ignore_z: bool = False,
        ignore_m: bool = False,
        action: NonFiniteAction = "drop",
        too_short: Literal["null", "keep", "empty"] = "null",
    ) -> pl.Expr:
        from geopolars.geo.filter_coordinates import filter_coordinates

        return filter_coordinates(
            self._expr,
            select_non_finite=select_non_finite,
            select_empty=select_empty,
            ignore_z=ignore_z,
            ignore_m=ignore_m,
            action=action,
            too_short=too_short,
        )

    def fix_out_of_bounds(
        self,
        *,
        crs_bounds: CrsBounds = "crs",
        action: OutOfBoundsAction = "drop",
        too_short: Literal["null", "keep", "empty"] = "null",
    ) -> pl.Expr:
        from geopolars.geo.fix_out_of_bounds import fix_out_of_bounds

        return fix_out_of_bounds(
            self._expr, crs_bounds=crs_bounds, action=action, too_short=too_short
        )

    def is_valid_geoarrow(
        self,
        *,
        out_of_bounds: OutOfBounds | None = None,
        check_null_coordinates: bool = True,
        check_finite_coordinates: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.is_valid import is_valid_geoarrow

        return is_valid_geoarrow(
            self._expr,
            out_of_bounds=out_of_bounds,
            check_null_coordinates=check_null_coordinates,
            check_finite_coordinates=check_finite_coordinates,
        )

    def null_out_invalid_geoarrow(
        self,
        *,
        out_of_bounds: OutOfBounds | None = None,
        check_null_coordinates: bool = True,
        check_finite_coordinates: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.is_valid import null_out_invalid_geoarrow

        return null_out_invalid_geoarrow(
            self._expr,
            out_of_bounds=out_of_bounds,
            check_null_coordinates=check_null_coordinates,
            check_finite_coordinates=check_finite_coordinates,
        )

    def is_valid_ogc(
        self,
        *,
        out_of_bounds: OutOfBounds | None = None,
        check_null_coordinates: bool = True,
        check_ring_self_intersect: bool = True,
        check_polygon_self_overlap: bool = True,
        check_multi_polygon_self_overlap: bool = True,
    ) -> pl.Expr:
        from geopolars.geo.is_valid import is_valid_ogc

        return is_valid_ogc(
            self._expr,
            out_of_bounds=out_of_bounds,
            check_null_coordinates=check_null_coordinates,
            check_ring_self_intersect=check_ring_self_intersect,
            check_polygon_self_overlap=check_polygon_self_overlap,
            check_multi_polygon_self_overlap=check_multi_polygon_self_overlap,
        )

    def null_out_invalid_ogc(
        self,
        *,
        out_of_bounds: OutOfBounds | None = None,
        check_null_coordinates: bool = True,
        check_ring_self_intersect: bool = True,
        check_polygon_self_overlap: bool = True,
        check_multi_polygon_self_overlap: bool = True,
    ) -> pl.Expr:
        from geopolars.geo.is_valid import null_out_invalid_ogc

        return null_out_invalid_ogc(
            self._expr,
            out_of_bounds=out_of_bounds,
            check_null_coordinates=check_null_coordinates,
            check_ring_self_intersect=check_ring_self_intersect,
            check_polygon_self_overlap=check_polygon_self_overlap,
            check_multi_polygon_self_overlap=check_multi_polygon_self_overlap,
        )

    def has_repeats_in_succession(
        self,
        *,
        abs_tol: float | None = None,
        rel_tol: float | None = None,
        unique_z: bool = True,
        unique_m: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.has_repeats_in_succession import has_repeats_in_succession

        return has_repeats_in_succession(
            self._expr,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            unique_z=unique_z,
            unique_m=unique_m,
        )

    def remove_repeats_in_succession(
        self,
        *,
        abs_tol: float | None = None,
        rel_tol: float | None = None,
        unique_z: bool = True,
        unique_m: bool = False,
        collapsed_lines: Literal["keep_ends", "single_vertex"] = "keep_ends",
        too_short: Literal["null", "keep", "empty"] = "null",
    ) -> pl.Expr:
        from geopolars.geo.remove_repeats_in_succession import (
            remove_repeats_in_succession,
        )

        return remove_repeats_in_succession(
            self._expr,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            unique_z=unique_z,
            unique_m=unique_m,
            collapsed_lines=collapsed_lines,
            too_short=too_short,
        )

    def distance(
        self,
        other: IntoExprColumn,
        *,
        method: Literal["planar", "geodesic"] | None = None,
    ) -> pl.Expr:
        from geopolars.geo.distance import distance

        return distance(self._expr, other, method=method)

    def length(
        self,
        *,
        method: Literal["planar", "geodesic"] | None = None,
        allow_non_lines: bool = False,
        ignore_nan: bool = False,
    ) -> pl.Expr:
        from geopolars.geo.length import length

        return length(
            self._expr,
            method=method,
            allow_non_lines=allow_non_lines,
            ignore_nan=ignore_nan,
        )

    def area(self, *, allow_non_polygons: bool = False) -> pl.Expr:
        from geopolars.geo.area import area

        return area(self._expr, allow_non_polygons=allow_non_polygons)

    def area_planar(self, *, allow_non_polygons: bool = False) -> pl.Expr:
        from geopolars.geo.area import area_planar

        return area_planar(self._expr, allow_non_polygons=allow_non_polygons)

    def area_geodesic(self, *, allow_non_polygons: bool = False) -> pl.Expr:
        from geopolars.geo.area import area_geodesic

        return area_geodesic(self._expr, allow_non_polygons=allow_non_polygons)

    def is_empty(self, *, check_nested: bool = False) -> pl.Expr:
        from geopolars.geo.is_empty import is_empty

        return is_empty(self._expr, check_nested=check_nested)

    def count_coordinates(self, *, include_empty: bool = False) -> pl.Expr:
        from geopolars.geo.count_coordinates import count_coordinates

        return count_coordinates(self._expr, include_empty=include_empty)

    def bounds(self, *, margin_x: float = 0.0, margin_y: float = 0.0) -> pl.Expr:
        from geopolars.geo.bounds import bounds

        return bounds(self._expr, margin_x=margin_x, margin_y=margin_y)

    def total_bounds(self, *, ignore_nulls: bool = False) -> pl.Expr:
        from geopolars.geo.bounds import total_bounds

        return total_bounds(self._expr, ignore_nulls=ignore_nulls)

    def to_polygon(self) -> pl.Expr:
        from geopolars.geo.to_polygon import to_polygon

        return to_polygon(self._expr)

    def wrap_longitude(
        self, *, start: float | None = None, skip_crossing: bool = False
    ) -> pl.Expr:
        from geopolars.geo.wrap_longitude import wrap_longitude

        return wrap_longitude(self._expr, start=start, skip_crossing=skip_crossing)

    def x(self) -> pl.Expr:
        return coordinates.x(self._expr)

    def y(self) -> pl.Expr:
        return coordinates.y(self._expr)

    def z(self) -> pl.Expr:
        return coordinates.z(self._expr)

    def m(self) -> pl.Expr:
        return coordinates.m(self._expr)

    def force_2d(self) -> pl.Expr:
        return force_dimension.force_2d(self._expr)

    def force_3d(self) -> pl.Expr:
        return force_dimension.force_3d(self._expr)

    def without_m(self) -> pl.Expr:
        return force_dimension.without_m(self._expr)

    def force_m(self) -> pl.Expr:
        return force_dimension.force_m(self._expr)

    def to_wkb(self) -> pl.Expr:
        return wkb.to_wkb(self._expr)

    def from_wkb(self, dtype: type[GeoArrowType], *, crs: str | None = None) -> pl.Expr:
        return wkb.from_wkb(self._expr, dtype, crs=crs)

    def to_wkt(self) -> pl.Expr:
        return wkt.to_wkt(self._expr)

    def from_wkt(self, dtype: type[GeoArrowType], *, crs: str | None = None) -> pl.Expr:
        return wkt.from_wkt(self._expr, dtype, crs=crs)
