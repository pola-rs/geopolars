"""The area a geometry encloses, on the plane or along the ellipsoid of its CRS.

Which of the two is picked by default
is described under "Geographic and projected" in `docs/source/user-guide/geo/concepts/crs.md`.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._typing import DistanceMethod, EarlyCoordCheck, OnRowError
from geopolars._utils import LIB, validate_one_of
from geopolars.datatypes import MultiPolygonType, PolygonType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _encloses_nothing(
    column: pl.Expr, geometry: GeoArrowType, *, allow_non_polygons: bool
) -> pl.Expr | None:
    """Only polygons and multipolygons enclose anything: `None` for those.
    Other geometries are refused, or given `0.0` with `allow_non_polygons`."""
    if isinstance(geometry, (PolygonType, MultiPolygonType)):
        return None
    if not allow_non_polygons:
        msg = (
            f"area does not accept a `{geometry._extension_name}` column, got: {geometry!r};"
            " pass `allow_non_polygons=True` to give it an area of 0.0"
        )
        raise TypeError(msg)
    return pl.when(column.is_not_null()).then(pl.lit(0.0, dtype=pl.Float64))


def _auto_area_method(geometry: GeoArrowType) -> Literal["planar", "geodesic"]:
    """Planar only without a CRS or in an equal-area one, whose plane keeps areas."""
    if not geometry._declares_crs() or geometry._is_equal_area():
        return "planar"
    return "geodesic"


def _area(
    column: pl.Expr,
    geometry: GeoArrowType,
    *,
    distance_method: DistanceMethod,
    allow_non_polygons: bool,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> pl.Expr:
    if distance_method == "auto":
        distance_method = _auto_area_method(geometry)
    elif distance_method == "geodesic" and not geometry._declares_crs():
        msg = (
            "a geodesic area needs a CRS to measure on, but the geometry declares none"
        )
        raise TypeError(msg)
    zero = _encloses_nothing(column, geometry, allow_non_polygons=allow_non_polygons)
    if zero is not None:
        return zero
    # Whether PROJ can use the CRS is for the kernel to check.
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="area",
        kwargs={
            "distance_method": distance_method,
            "on_row_error": on_row_error,
            "early_coord_check": early_coord_check,
        },
        is_elementwise=True,
    )


@doc()
def area(
    geometry: IntoExprColumn,
    *,
    distance_method: DistanceMethod = "auto",
    allow_non_polygons: bool = False,
    on_row_error: OnRowError = "raise",
    early_coord_check: EarlyCoordCheck = "finite",
) -> pl.Expr:
    """The area a geometry encloses, as an `f64`.

    ===================  =========================  =====================
    `distance_method`    measured                   in
    ===================  =========================  =====================
    `"auto"`             depends on CRS             -
    `"planar"`           shoelace formula           coordinate units²
    `"geodesic"`         along the CRS's ellipsoid  CRS units²
    ===================  =========================  =====================


    **Planar** calculates the area directly based on the values of the coordinate data.
    This is very useful for data where the position is already in meters,
    (such as with equal-area projections).
    Because it can ignore the CRS, it is also very fast to calculate.
    For geographic CRSes, where the coordinates are longitude/latitude,
    this operation gives square degrees, which is most likely not what you want though!
    **Geodesic** takes the coordinates to the longitude/latitude
    the CRS is defined on (for a projected CRS, the one it projects from;
    no datum is shifted),
    and measures every ring as the region its geodesic edges enclose
    on that CRS's own ellipsoid, as `distance` measures two points.
    The result is in the unit of the CRS's `x` and `y` squared
    (such as square US survey feet for EPSG:2263),
    or in square metres when those are degrees.


    `"geodesic"` refuses a geometry without a CRS while the plan is built.
    `"geodesic"` and `"auto"` also refuse a CRS that is not defined on longitude/latitude.

    The default, `distance_method="auto"`, measures a geometry without a CRS,
    or in an equal-area CRS (whose plane keeps areas), `"planar"`.
    A geometry in any other CRS is measured `"geodesic"`,
    including a projected one such as a UTM zone.
    This differs from `"auto"` in `distance`, which measures every projected CRS on its plane.

    If a ring of the polygon spans more than half the ellipsoid,
    it takes the smaller side of the enclosed surface.

    Winding order of the holes does not matter.
    Holes are subtracted from the enclosed volume.
    A multipolygon gets the sum of the areas of its polygons.

    Only the two-dimensional part is considered: `z` and `m` are ignored,
    so a a 3D polygon returns the area of its footprint
    (its shadow on the x/y plane), not the area of its surface in 3D.
    a tilted polygon 3D polygon gets less than its true area, and a vertical one gets `0.0`.

    An empty polygon encloses nothing, so it gets `0.0`.
    Every other geometry (points, linestrings and their multi forms)
    is refused while the plan is built,
    unless `allow_non_polygons` is set: then each of them gets `0.0`
    (and a null stays null).

    =======================  ============
    in                       out
    =======================  ============
    `PointXY[ZM]`            refused
    `LineStringXY[ZM]`       refused
    `PolygonXY[ZM]`          `0.0` and up
    `MultiPointXY[ZM]`       refused
    `MultiLineStringXY[ZM]`  refused
    `MultiPolygonXY[ZM]`     `0.0` and up
    =======================  ============

    A self-intersecting ring is not valid by OGC (see `is_valid_ogc`), and the parts of it that
    wind the other way will cancel rather than add.

    {on_row_error}
    A row fails when PROJ cannot take one of its vertices to longitude/latitude (only for `"geodesic"`),
    or when one of its x or y is NaN or infinite, unless `early_coord_check="off"`.

    {early_coord_check}

    .. code-block:: python

        df.select(geo.area("parcel"))
        df.select(geo.area("parcel", distance_method="planar"))
    """
    validate_one_of("distance_method", distance_method, DistanceMethod)
    validate_one_of("on_row_error", on_row_error, OnRowError)
    validate_one_of("early_coord_check", early_coord_check, EarlyCoordCheck)
    return on_geometry(
        geometry,
        partial(
            _area,
            distance_method=distance_method,
            allow_non_polygons=allow_non_polygons,
            on_row_error=on_row_error,
            early_coord_check=early_coord_check,
        ),
    )
