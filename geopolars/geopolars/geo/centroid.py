"""The centroid of a geometry: the centre of its length or area.
Note that this is not the mean coordinate, which weighs every vertex the same!

A native kernel (A-tier) does either in one pass over the storage:
it weighs every segment of a linestring by its length,
and applies the shoelace formula over a polygon's planar area.
A multilinestring or multipolygon gets the centroid of its parts together,
each weighted by its length or area.
A (multi)point has neither length nor area,
so its centroid is its mean coordinate (C-tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import MultiPointType, PointType
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.mean_coordinate import _mean_coordinate

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _centroid(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    """The centroid of one geometry type, per row."""
    # Every point weighs the same, so the centroid is the mean of the points.
    if isinstance(geometry, PointType | MultiPointType):
        return _mean_coordinate(column, geometry)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="centroid",
        is_elementwise=True,
    )


def centroid(geometry: IntoExprColumn) -> pl.Expr:
    """The centroid of a polygon (centre of area) or linestring (centre of length).

    A polygon's centroid is measured on the plane the coordinates lie in,
    even when they carry a CRS, and the CRS is carried over to the result.
    Holes are taken out.
    Its `x` and `y` are the centroid of its footprint (its shadow on the x/y plane),
    and its `z` and `m` are averaged over that footprint's area.
    For a flat 3D polygon, that puts the centroid on the polygon,
    right above the centroid of its footprint.
    The result keeps the polygon's dimension.
    A vertical polygon has a footprint that encloses nothing, so it has no centroid.

    A linestring's centroid is the midpoint of each of its segments,
    weighted by that segment's length.
    Fully three-dimensional if `z`.
    If the line has an `m`, it is averaged.
    The result keeps the line's dimension.
    A null linestring, an empty one,
    and one without length (a single vertex, or all on one spot) have no centroid.

    A null geometry, an empty polygon, and a polygon that encloses nothing
    (all of its vertices on a line) have no centroid.

    A multilinestring or multipolygon gets a single centroid:
    that of each part (measured as for a lone linestring or polygon),
    weighted by its length or area.
    A part without length or area adds nothing.
    A null geometry, an empty one,
    and one whose parts have no length or area at all have no centroid.

    A point is its own centroid.
    A multipoint gets a single centroid, which is the mean of its points
    (see `mean_coordinate`).
    A null or empty multipoint has no centroid.

    =======================  =============
    in                       out
    =======================  =============
    `PointXY[ZM]`            `PointXY[ZM]`
    `LineStringXY[ZM]`       `PointXY[ZM]`
    `PolygonXY[ZM]`          `PointXY[ZM]`
    `MultiPointXY[ZM]`       `PointXY[ZM]`
    `MultiLineStringXY[ZM]`  `PointXY[ZM]`
    `MultiPolygonXY[ZM]`     `PointXY[ZM]`
    =======================  =============

    .. code-block:: python

        df.select(geo.centroid("parcel"), geo.centroid("road"))
    """
    return on_geometry(geometry, _centroid)
