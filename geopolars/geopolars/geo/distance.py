"""How far apart two points are, on the plane or along the ellipsoid of their CRS.

Which of the two is picked by default, and why,
is described under "Geographic and projected" in `docs/source/user-guide/geo/concepts/crs.md`.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import PointType
from geopolars.geo._dispatch import on_geometry_pair

if TYPE_CHECKING:
    from geopolars._typing import DistanceMethod, IntoExprColumn
    from geopolars.datatypes import GeoArrowType


# TODO: add expr for things with multiple points, get the 'closest distance'.
# with flag to also interpolate between points / line?


def _auto_distance_method(dtype: GeoArrowType) -> Literal["planar", "geodesic"]:
    """Planar only without a CRS or in a projected one.
    Any other CRS, such as a geocentric one, goes geodesic, which refuses it if it has no ellipsoid.
    """
    if not dtype._declares_crs() or dtype._is_projected():
        return "planar"
    return "geodesic"


def _distance(
    a: pl.Expr,
    a_dtype: GeoArrowType,
    b: pl.Expr,
    b_dtype: GeoArrowType,
    *,
    distance_method: DistanceMethod,
) -> pl.Expr:
    # TODO: for other shapes, calculate centroid, and take distance on that.
    if not (isinstance(a_dtype, PointType) and isinstance(b_dtype, PointType)):
        msg = (
            "a distance is measured between two `geoarrow.point` columns, "
            f"got: {a_dtype!r} and {b_dtype!r}"
        )
        raise TypeError(msg)

    # Asking one side is enough, as the kernel refuses points that do not share a CRS.
    crs_side = a_dtype if a_dtype._declares_crs() else b_dtype
    if distance_method == "auto":
        distance_method = _auto_distance_method(crs_side)
    elif distance_method == "geodesic" and not crs_side._declares_crs():
        msg = "a geodesic distance needs a CRS to measure on, but neither point declares one"
        raise TypeError(msg)

    # Whether both declare the same CRS, and whether PROJ can use it,
    # is for the kernel to check while resolving the schema.
    # The kernel counts the height itself, when both points carry one.
    return register_plugin_function(
        plugin_path=LIB,
        args=[a, b],
        function_name="distance",
        kwargs={"distance_method": distance_method},
        is_elementwise=True,
    )


def distance(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    distance_method: DistanceMethod = "auto",
) -> pl.Expr:
    """How far apart two points are, row by row, as an `f64`.

    =================  =========================  ================
    `distance_method`  measured                   in
    =================  =========================  ================
    `"planar"`         Pythagorean                coordinate units
    `"geodesic"`       along the CRS's ellipsoid  CRS units
    =================  =========================  ================

    The default, `distance_method="auto"`, measures points in a projected CRS, or without one, `"planar"`,
    and points in any other CRS `"geodesic"`.
    A CRS that is not defined on longitude/latitude, such as a geocentric one, has no ellipsoid
    to measure along, so `"geodesic"` and `"auto"` refuse it while the plan is built.
    Ask for `"planar"` to measure it on its `x` and `y` anyway.
    Planar is close to geodesic in a projected CRS made for measuring, such as a UTM zone,
    but Web Mercator (EPSG:3857) stretches distances by 1/cos(latitude):
    pass `"geodesic"` for such a CRS.
    With any method, points in different CRSs, or a CRS on only one of them, are refused:
    reproject one onto the other with `to_crs` first.
    **Planar** ignores the CRS: for longitude/latitude it comes out in degrees,
    which are not the same length east-west as north-south.
    **Geodesic** takes the points to the longitude/latitude the CRS is defined on
    (for a projected CRS, the one it projects from; no datum is shifted)
    and measures along the geodesic between them, on that CRS's own ellipsoid.
    The result is in the unit of the CRS's `x` and `y` (such as US survey feet for EPSG:2263),
    or in metres when those are degrees.
    Points without a CRS have no ellipsoid, so `"geodesic"` refuses them.

    When both points have a `z`, the difference in height counts too:
    the distance is `sqrt(d² + Δz²)`, with `d` the distance above.
    With `"geodesic"`, `z` is taken to be in the same unit as the result,
    and the geodesic is measured at the surface of the ellipsoid
    (a few parts per million short at aircraft heights).
    If only one of the points has a `z`, there is no height difference to count,
    and they are measured as if neither had one.
    `m` is always ignored.
    An empty point gives a NaN distance.
    Either side may be a single point, in which case its distance to all the other points is measured.

    .. code-block:: python

        df.select(geo.distance("home", "work"))
        df.select(geo.distance("home", "work", distance_method="planar"))
    """
    return on_geometry_pair(a, b, partial(_distance, distance_method=distance_method))
