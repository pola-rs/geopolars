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
) -> pl.Expr:
    if distance_method == "auto":
        distance_method = _area_auto_method(geometry)
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
        },
        is_elementwise=True,
    )


def _area_auto_method(geometry: GeoArrowType) -> Literal["planar", "geodesic"]:
    if not geometry._declares_crs() or geometry._is_equal_area():
        return "planar"
    return "geodesic"


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
) -> pl.Expr:
    """The area a geometry encloses, as an `f64`.

    ===================  =========================  ==========================
    `distance_method`    measured                   in
    ===================  =========================  ==========================
    `"auto"`             depends on CRS             -
    `"planar"`           shoelace formula           coordinate units²
    `"geodesic"`         along the CRS's ellipsoid  CRS units², m² for degrees
    ===================  =========================  ==========================


    **Planar** measures the area straight from the coordinates.
    Does not need to do any projection, which makes it very fast.
    It does not need to check the CRS at all, unless `on_out_of_bounds` asks for bounds checking.
    This does mean its area is only right in an equal-area CRS.
    In any other projected CRS (a UTM zone included),
    the error depends on the projection area distortion.
    In a geographic CRS, the coordinates are longitude/latitude,
    which means that the planar area gives a value in square degrees.
    For most use cases, this is probably not what you want.

    **Geodesic** takes the CRS into consideration.
    It measures every ring as the region its geodesic edges enclose on the CRS's own ellipsoid.
    The result is in the unit of the CRS's `x` and `y` squared
    (such as square US survey feet for EPSG:2263),
    or in square metres when those are degrees.

    `"geodesic"` refuses a geometry without a CRS.

    {crs_without_ellipsoid}

    The default, `distance_method="auto"`, attempts to get the meaningful value in the cheapest way possible.
    If the geometry has no CRS, or an equal-area one, it will use the `"planar"` method.
    Geometry in any other CRS is measured using `"geodesic"` (including a projected one such as a UTM zone).
    Note that this differs from `"auto"` in `distance`, which measures every projected CRS on its plane.
    This is because distance preservation does not necessarily imply area preservation.


    A hole is always subtracted from the exterior, whichever way their rings wind.
    A multipolygon gets the sum of the areas of its polygons.

    Only `x` and `y` are measured, and `z` and `m` are ignored.
    A 3D polygon therefore gets the area of its footprint
    (its shadow on the x/y plane), not the area of its surface in 3D.
    A tilted 3D polygon gets less than its true area, and a vertical one gets `0.0`.

    An empty polygon encloses nothing, so it gets `0.0`.
    Every other geometry (points, linestrings, and their multi forms)
    is refused while the plan is built.
    `allow_non_polygons=True` gives each of them `0.0` instead.

    =======================  ============
    in                       out
    =======================  ============
    `PointXY[ZM]`            refused
    `LineStringXY[ZM]`       refused
    `PolygonXY[ZM]`          `f64`
    `MultiPointXY[ZM]`       refused
    `MultiLineStringXY[ZM]`  refused
    `MultiPolygonXY[ZM]`     `f64`
    =======================  ============


    A closed ring of fewer than 4 vertices cannot span a surface,
    so it encloses nothing and gets `0.0`.

    A NaN or infinite coordinate never gives a finite area.
    This takes precedence over the length of the ring.
    (One edge case: with `early_coord_check="off"` and `on_out_of_bounds="ignore"`,
    PROJ can take an infinite coordinate to a finite longitude/latitude when measuring geodesic,
    such as `y=-inf` in Web Mercator to the south pole, which then gives a finite area.)
    An empty point in a ring is a NaN coordinate too, so it gives NaN.
    Use `.fill_nan(None)` on the area to make these null instead,
    or `null_out_invalid_ogc` on the polygons before measuring them.


    A polygon that is not valid by OGC can lead to unexpected behaviour such as negative areas.
    Examples where this happens are:
    polygons with an empty exterior and a non-empty hole,
    polygons with holes that are larger than their exterior,
    or polygons with duplicate holes.
    In a multipolygon, a negative value is summed normally.

    Parts where the winding order changes inside of a ring (such as in a self-intersecting bow-tie)
    way cancel rather than add.

    This operation always uses the smallest area that can be enclosed by the polygon.
    A polygon that covers more than half the ellipsoid is not supported,
    as described under "Polygon edges" in `docs/source/user-guide/geo/concepts/geometry-types.md`.
    This means that a valid polygon whose geodesic exterior covers more than half the ellipsoid
    can still give a negative area, as its exterior gets the smaller region.

    {on_row_error}
    A geodesic row fails when PROJ cannot take one of its vertices to longitude/latitude.
    By default, it also fails when it holds a coordinate outside the bounds of its CRS (see below).

    {early_coord_check}
    Only `"geodesic"` runs the check, as the planar formula costs no more than the check does.
    A geodesic row that fails it gets NaN, without being measured.
    Measured planar, a NaN or infinite coordinate is simply propagated arithmetically.
    With a bounds check (see below), an infinite coordinate then counts as out of bounds instead.

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
        ),
    )
