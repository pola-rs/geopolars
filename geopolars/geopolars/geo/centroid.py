"""The centroid of a geometry, the centre of its length or area."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._typing import OnRowError
from geopolars._utils import LIB, validate_one_of
from geopolars.datatypes import MultiPointType, PointType
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.mean_coordinate import _mean_coordinate

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

CentroidMethod = Literal["footprint", "3d"]
CentroidAnchor = Literal["length_weighted_mean", "vertex_mean"]


def _centroid(
    column: pl.Expr,
    geometry: GeoArrowType,
    method: CentroidMethod,
    anchor: CentroidAnchor,
    on_row_error: OnRowError,
) -> pl.Expr:
    """The centroid of one geometry type, per row."""
    # Every point weighs the same, so the centroid is the mean of the points.
    if isinstance(geometry, PointType | MultiPointType):
        return _mean_coordinate(column, geometry, empty="empty_point")
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="centroid",
        is_elementwise=True,
        kwargs={"method": method, "anchor": anchor, "on_row_error": on_row_error},
    )


@doc()
def centroid(
    geometry: IntoExprColumn,
    *,
    method: CentroidMethod = "footprint",
    anchor: CentroidAnchor = "length_weighted_mean",
    on_row_error: OnRowError = "raise",
) -> pl.Expr:
    """The centroid of a polygon (centre of area) or linestring (centre of length).

    Every segment of a linestring weighs as much as it is long,
    and a polygon is weighted evenly over its area without its holes.
    The centroid is the average of every axis under those weights.
    It is therefore not the mean coordinate, which weighs every vertex the same
    (see `mean_coordinate`).
    Lengths and areas are measured on the plane the coordinates lie in,
    even when they carry a CRS.

    `method` says which axes measure a length or an area:

    - `"footprint"` is the default. It only measures `x` and `y`, as `area(distance_method="planar")` does.
      A geometry with a `z` is measured by its footprint (its shadow on the x/y plane),
      and its `z` is averaged over that footprint.
      For a flat polygon that is not vertical,
      this puts the centroid on the polygon's plane, right above the centroid of its footprint.
      A vertical linestring or polygon has no centroid,
      as its footprint has no length or area.
    - `"3d"` measures `x`, `y`, and `z`, as `length` does.
      A linestring's segments then count their difference in height too.
      A polygon is measured on its own plane,
      so that a vertical one has a centroid as well.
      A polygon that is not flat has no plane of its own.
      It is measured by its shadow on the plane that faces along its vector area instead
      (Newell's method), which gives the same plane for a flat one.

    Without a `z`, both methods give the same centroid.

    A polygon's area is split into triangles,
    which fan out from one point per ring.
    `anchor` picks that point.
    It makes no difference to where the centroid of a flat polygon lies.
    For a polygon that is not flat, it moves the `z` with `"footprint"`,
    and every axis with `"3d"`.
    It also moves an `m` that does not change linearly over the polygon.

    - `"length_weighted_mean"` is the default.
      It fans out from the centre of the ring's length, measured as `method` says.
      The centroid therefore does not change when a vertex is added on a straight edge,
      or when the ring starts at another vertex.
    - `"vertex_mean"` fans out from the mean of the ring's vertices.
      It is quicker to find, but it moves when a vertex is added on a straight edge,
      and the centroid with it.

    A linestring has no area to fan out over, so it ignores `anchor`.

    {measure}
    As it is not a position, it never adds to a length or area, whatever the `method`.
    It is averaged with the same weights as the positions,
    which gives the mean `m` over the length or area of the geometry.

    A multilinestring or multipolygon gets a single centroid.
    Its parts are weighted together by their length or area,
    measured as for a lone linestring or polygon.

    An empty geometry (one without any vertices, in any of its parts)
    gives the empty point.

    {on_row_error}
    A geometry with vertices but without length or area has no centroid,
    such as a linestring with all its vertices on one spot,
    or a polygon with all its vertices on a line.
    GEOS and Shapely fall back to the centroid of its length, or else to the mean of its vertices.
    This does not, as a centroid of another kind would go unnoticed.
    The error for it suggests `method="3d"` when the geometry has a `z`,
    as a vertical geometry only has length or area in 3D.
    There are two more cases without a centroid, which each have an error of their own.
    A geometry whose vertices all lie very close to the first
    (within about 1e-154 for a linestring, or 1e-103 for a polygon)
    has a length or area too small to be held by a float.
    A polygon whose holes take out more area than its exterior holds is not valid,
    and would have a negative area.

    A NaN or an infinity in a coordinate gives a centroid that is not finite,
    rather than a null.
    In an axis that measures the length or area, it makes every axis NaN.
    In any other axis (`m`, or `z` with `"footprint"`), it only changes that axis.
    For a multilinestring or multipolygon, this holds for the centroid of the whole,
    whichever part the coordinate is in.

    A point is its own centroid.
    A multipoint gets the mean of its points as its centroid (see `mean_coordinate`),
    whatever the `method` and `anchor`.

    The result keeps the geometry's dimension and CRS.

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
        df.select(geo.centroid("flight_path", method="3d"))
    """
    validate_one_of("method", method, CentroidMethod)
    validate_one_of("anchor", anchor, CentroidAnchor)
    validate_one_of("on_row_error", on_row_error, OnRowError)
    return on_geometry(
        geometry,
        lambda column, dtype: _centroid(column, dtype, method, anchor, on_row_error),
    )
