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
from geopolars._typing import (
    DistanceMethod,
    EarlyCoordCheck,
    OnOutOfBounds,
    OnRowError,
)
from geopolars._utils import LIB, validate_one_of
from geopolars.datatypes import MultiPolygonType, PolygonType
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.fix_out_of_bounds import CrsBounds, _bounds

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

NegativeArea = Literal["zero", "keep"]
ExteriorSide = Literal["fit_holes", "smaller"]


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
    on_out_of_bounds: OnOutOfBounds,
    crs_bounds: CrsBounds,
    negative_area: NegativeArea,
    exterior_side: ExteriorSide,
) -> pl.Expr:
    if distance_method == "auto":
        distance_method = _auto_area_method(geometry)
    elif distance_method == "geodesic" and not geometry._declares_crs():
        msg = (
            "a geodesic area needs a CRS to measure on, but the geometry declares none"
        )
        raise TypeError(msg)
    if on_out_of_bounds == "auto":
        on_out_of_bounds = "ignore" if distance_method == "planar" else "error"
    zero = _encloses_nothing(column, geometry, allow_non_polygons=allow_non_polygons)
    if zero is not None:
        return zero
    # PROJ is only asked for the bounds when they are checked.
    bounds = None if on_out_of_bounds == "ignore" else _bounds(geometry, crs_bounds)
    # Whether PROJ can measure in the CRS is for the kernel to check, while the plan is built.
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="area",
        kwargs={
            "distance_method": distance_method,
            "on_row_error": on_row_error,
            "early_coord_check": early_coord_check,
            "bounds": bounds,
            "on_out_of_bounds": on_out_of_bounds,
            "negative_area": negative_area,
            "exterior_side": exterior_side,
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
    on_out_of_bounds: OnOutOfBounds = "auto",
    crs_bounds: CrsBounds = "crs_wrapped_longitude",
    negative_area: NegativeArea = "zero",
    exterior_side: ExteriorSide = "fit_holes",
) -> pl.Expr:
    """The area a geometry encloses, as an `f64`.

    ===================  =========================  ==========================
    `distance_method`    measured                   in
    ===================  =========================  ==========================
    `"auto"`             depends on CRS             -
    `"planar"`           shoelace formula           coordinate units²
    `"geodesic"`         along the CRS's ellipsoid  CRS units², m² for degrees
    ===================  =========================  ==========================


    **Planar** calculates the area directly based on the values of the coordinate data.
    It only keeps areas in an equal-area CRS:
    in any other projected CRS (a UTM zone included),
    it is off by as much as the projection distorts areas.
    Because it can ignore the CRS, it is also very fast to calculate:
    unless `on_out_of_bounds` asks for it, it does not check the bounds of the CRS either.
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

    With `"geodesic"`, a ring encloses the smaller of the two regions it splits the ellipsoid into.
    With `exterior_side="fit_holes"` (default), an exterior whose holes take more than that
    gets the larger one instead, the only side they fit in.
    `exterior_side="smaller"` skips this, which is faster.
    `"planar"` ignores `exterior_side`.

    Which way a ring winds does not matter: a hole is always subtracted from the exterior.
    A multipolygon gets the sum of the areas of its polygons.

    Only the two-dimensional part is considered: `z` and `m` are ignored,
    so a 3D polygon gets the area of its footprint
    (its shadow on the x/y plane), not the area of its surface in 3D.
    A tilted 3D polygon gets less than its true area, and a vertical one gets `0.0`.

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

    A polygon that is not valid by OGC can come out below zero
    (think of polygons with an empty exterior but a hole, or with holes that enclose more than its exterior).
    So can a valid one with `exterior_side="smaller"`,
    whose geodesic exterior covers more than half the ellipsoid.
    `negative_area="zero"` gives such a polygon `0.0`, and `negative_area="keep"` its negative area.
    In a multipolygon, this applies to each individual polygon before they are summed.

    {on_row_error}
    With `"geodesic"`, a row holding a coordinate outside the bounds of its CRS fails by default (see below).
    PROJ failing to take one of its vertices to longitude/latitude also fails it, with `"geodesic"`.

    {early_coord_check}
    A row that fails the check gets NaN, without being measured.

    {on_out_of_bounds}

    .. code-block:: python

        df.select(geo.area("parcel"))
        df.select(geo.area("parcel", distance_method="planar"))
    """
    validate_one_of("distance_method", distance_method, DistanceMethod)
    validate_one_of("on_row_error", on_row_error, OnRowError)
    validate_one_of("early_coord_check", early_coord_check, EarlyCoordCheck)
    validate_one_of("on_out_of_bounds", on_out_of_bounds, OnOutOfBounds)
    validate_one_of("crs_bounds", crs_bounds, CrsBounds)
    validate_one_of("negative_area", negative_area, NegativeArea)
    validate_one_of("exterior_side", exterior_side, ExteriorSide)
    return on_geometry(
        geometry,
        partial(
            _area,
            distance_method=distance_method,
            allow_non_polygons=allow_non_polygons,
            on_row_error=on_row_error,
            early_coord_check=early_coord_check,
            on_out_of_bounds=on_out_of_bounds,
            crs_bounds=crs_bounds,
            negative_area=negative_area,
            exterior_side=exterior_side,
        ),
    )
