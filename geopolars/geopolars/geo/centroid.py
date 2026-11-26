"""The centroid of a geometry, the centre of its length or area."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._typing import DistanceMethod, OnRowError
from geopolars._utils import LIB, validate_one_of
from geopolars.datatypes import (
    MultiPointType,
    MultiPolygonType,
    PointType,
    PolygonType,
)
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.area import _area_auto_method
from geopolars.geo.distance import _auto_distance_method
from geopolars.geo.mean_coordinate import _mean_coordinate

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

CentroidWeighing = Literal["footprint", "3d"]
CentroidAnchor = Literal["length_weighted_mean", "vertex_mean"]


def _centroid(
    column: pl.Expr,
    geometry: GeoArrowType,
    weighing_method: CentroidWeighing,
    anchor: CentroidAnchor,
    distance_method: DistanceMethod,
    on_row_error: OnRowError,
) -> pl.Expr:
    """The centroid of one geometry type, per row."""
    _centroid_refuse_geodesic(geometry, distance_method)
    # Every point weighs the same, so the centroid is the mean of the points.
    if isinstance(geometry, PointType | MultiPointType):
        return _mean_coordinate(column, geometry, empty="empty_point")
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="centroid",
        is_elementwise=True,
        kwargs={
            "weighing_method": weighing_method,
            "anchor": anchor,
            "on_row_error": on_row_error,
        },
    )


def _centroid_refuse_geodesic(
    geometry: GeoArrowType, distance_method: DistanceMethod
) -> None:
    # A point is its own centroid, however it is measured.
    if distance_method == "planar" or isinstance(geometry, PointType):
        return
    if distance_method == "auto":
        if isinstance(geometry, PolygonType | MultiPolygonType):
            if _area_auto_method(geometry) == "planar":
                return
            like, crs_kind = "area", "equal-area"
        else:
            if _auto_distance_method(geometry) == "planar":
                return
            like, crs_kind = "length", "projected"
        reason = (
            f'`distance_method="auto"` measures {geometry!r} geodesically'
            f" as `{like}` would, since its CRS is not {crs_kind}"
        )
    else:
        reason = '`distance_method="geodesic"` was asked for'
    msg = (
        f"{reason}, but centroid cannot measure geodesically yet;"
        ' pass `distance_method="planar"` to measure the coordinates as they are,'
        " or reproject the geometry first"
    )
    raise NotImplementedError(msg)


@doc()
def centroid(
    geometry: IntoExprColumn,
    *,
    weighing_method: CentroidWeighing = "footprint",
    anchor: CentroidAnchor = "length_weighted_mean",
    distance_method: DistanceMethod = "planar",
    on_row_error: OnRowError = "raise",
) -> pl.Expr:
    """The centroid is the 'middle' of a geometry, represented as a weighted average of every axis.
    What exactly this is depends on the geometry.
    This weighing makes it different from `mean_coordinate`.
    The result is always a `PointXY[ZM]`.
    The result inherits the geometry's dimension and CRS.
    TODO: ^ shared doc

    For a polygon, this gives the centre of area, weighted evenly over its area without its holes.
    For a linestring, it gives the centre of length, where every segment weighs as much as it is long.

    `distance_method` says how lengths and areas are measured
    (see :ref:`distance_method <arg-distance_method>` on the Common arguments page),
    as this can affect the weighing.
    `"auto"` picks the method `length` would for a linestring, and the one `area` would for a polygon.
    Only `"planar"` is supported for now, which is why it is the default.

    `weighing_method` determines which axes are considered in the weighing process.
    It only has an effect if the data has a `z` dimension.

    - `"footprint"` is the default. It only measures `x` and `y`, as `area(distance_method="planar")` does.
      A geometry with a `z` is therefore measured by its footprint (its shadow on the x/y plane),
      with its `z` simply averaged over it.
      For a flat polygon that is not vertical,
      this puts the centroid on the polygon's plane, right above the centroid of its footprint.
      A vertical linestring or polygon calculated using the `"footprint"` method therefore has no centroid,
      as its footprint has no length or area.
      This method is cheaper, and more useful if your data represents a surface that happens to have elevation data.
    - `"3d"` also considers `z` in the weighing process.
      A linestring's segments then count their difference in height too.
      The calculation uses Newell's method.
      Here, a non-flat polygon is measured by its shadow on the plane facing along its vector area.
      With this method, a vertical linestring or polygon will have a centroid too.
      This is more expensive, but more representative if your data represents a sampled data set with height information.

    A polygon's area is split into triangles that fan out from one point per ring.
    `anchor` sets from where the fanning of a polygon's area is done.
    If the polygon does not lie on a plane, it moves the `z` with `"footprint"`,
    and every axis with `"3d"`.
    It has no effect if the polygon is flat.
    It also moves an `m` that does not change linearly over the polygon.
    A linestring has no area to fan out over, so it ignores `anchor`.

    - `"length_weighted_mean"` is the default.
      It fans out from the centre of the ring's length (measured as `weighing_method` says).
      The centroid therefore does not change when a vertex is added on a straight edge,
      or when the ring starts at another vertex.
    - `"vertex_mean"` fans out from the mean of the ring's vertices.
      It is quicker to find, but it moves when a vertex is added on a straight edge,
      and so does the centroid of a polygon that is not flat.

    {measure}
    As it is not a position, it never contributes to the weights.
    The resulting `m` of the centroid is the geometry's weighted mean value.

    A multilinestring or multipolygon gets a single centroid.
    Its calculation is done per part, and then compounded.

    An empty geometry (one without any vertices, in any of its parts)
    gives the empty point as its centroid.

    {on_row_error}
    If the geometry has no length or no area, it will have no centroid, such as:

    - a linestring with all its vertices on one spot.
    - a polygon with all its vertices exactly on a line.
      A polygon whose vertices lie on a line only up to rounding
      (such as points computed as `(x, 3 * x)`) can underflow.
      That centroid can lie far outside the polygon.
      TODO: make it automatically fall back to line weight in this rare case, and otherwise to vertex mean.
    - a geometry whose vertices all lie within about 1e-162 of each other,
      because it gets a tiny length or area.
      A slightly larger geometry (within about 1e-154 for a linestring,
      or 1e-103 for a polygon) has a centroid, but it loses precision relative to the size of the geometry.

    An invalid polygon that has a negative area due to oversized holes raises an error too.

    A NaN or an infinity propagates this value through arithmetically,
    resulting in a centroid that itself is also non-finite.
    If the axis that this value is used in is part of the weighing process (see `weighing_method`),
    the entire centroid will be NaN as the summed weight is now non-finite.
    If it is not used to weigh (`m`, or `z` with `"footprint"`), it only changes that axis.

    A point is its own centroid, whatever the `distance_method`.
    A multipoint gets the mean of its points as its centroid (see `mean_coordinate`),
    whatever the `weighing_method` and `anchor`.
    `distance_method="auto"` picks the method for it that `length` would for a linestring.

    .. code-block:: python

        df.select(geo.centroid("parcel"), geo.centroid("road"))
        df.select(geo.centroid("flight_path", weighing_method="3d"))
    """
    validate_one_of("weighing_method", weighing_method, CentroidWeighing)
    validate_one_of("anchor", anchor, CentroidAnchor)
    validate_one_of("distance_method", distance_method, DistanceMethod)
    validate_one_of("on_row_error", on_row_error, OnRowError)
    return on_geometry(
        geometry,
        lambda column, dtype: _centroid(
            column, dtype, weighing_method, anchor, distance_method, on_row_error
        ),
    )
