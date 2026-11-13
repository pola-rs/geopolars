"""The area a geometry encloses.

`area_planar` measures on the flat plane the coordinates lie in,
in whatever units those are squared.
`area_geodesic` measures along the ellipsoid of the CRS, in the unit of the CRS squared
(square metres for a CRS in longitude/latitude).
`area` picks between them by the CRS.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
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


def _planar(
    column: pl.Expr, geometry: GeoArrowType, *, allow_non_polygons: bool
) -> pl.Expr:
    zero = _encloses_nothing(column, geometry, allow_non_polygons=allow_non_polygons)
    if zero is not None:
        return zero
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="area_planar",
        is_elementwise=True,
    )


def _geodesic(
    column: pl.Expr, geometry: GeoArrowType, *, allow_non_polygons: bool
) -> pl.Expr:
    # Whether PROJ can use the CRS is for the kernel to check.
    if not geometry._declares_crs():
        msg = "area_geodesic needs a CRS to measure on, but the geometry declares none"
        raise TypeError(msg)
    zero = _encloses_nothing(column, geometry, allow_non_polygons=allow_non_polygons)
    if zero is not None:
        return zero
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="area_geodesic",
        is_elementwise=True,
    )


def area_planar(
    geometry: IntoExprColumn, *, allow_non_polygons: bool = False
) -> pl.Expr:
    """The area a geometry encloses on the plane its coordinates lie in, as an `f64`.

    A CRS is ignored: for longitude/latitude that gives square degrees,
    which is rarely what you want (see `area_geodesic`).

    The area is computed with the shoelace formula, in the unit of the coordinates squared.
    Only a polygon actually encloses something.
    This also considers its holes.
    Which way a ring winds does not matter: a hole is always subtracted from the exterior.
    A multipolygon gets the sum of the areas of its polygons.
    Only the two-dimensional part is considered: 'z' and 'm' are ignored.
    For a 3D polygon that gives the area of its footprint
    (its shadow on the x/y plane), not the area of its surface in 3D:
    a tilted polygon gets less than its true area, and a vertical one gets `0.0`.

    An empty polygon encloses nothing, so it gets `0.0`.
    Every other geometry (points, linestrings and their multi forms)
    is refused while the plan is built,
    unless `allow_non_polygons` is set: then each of them gets `0.0`
    (and a null stays null).

    | in | out |
    | - | - |
    | `PointXY[ZM]` | refused |
    | `LineStringXY[ZM]` | refused |
    | `PolygonXY[ZM]` | `0.0` and up |
    | `MultiPointXY[ZM]` | refused |
    | `MultiLineStringXY[ZM]` | refused |
    | `MultiPolygonXY[ZM]` | `0.0` and up |

    A self-intersecting ring is not valid by OGC (see `is_valid_ogc`), and the parts of it that
    wind the other way will cancel rather than add.

    ```python
    df.select(geo.area_planar("parcel"))
    ```
    """
    return on_geometry(
        geometry, partial(_planar, allow_non_polygons=allow_non_polygons)
    )


def area_geodesic(
    geometry: IntoExprColumn, *, allow_non_polygons: bool = False
) -> pl.Expr:
    """The area a geometry encloses along the ellipsoid of its CRS, as an `f64`.

    The coordinates are taken to the longitude/latitude
    the CRS is defined on (for a projected CRS, the one it projects from;
    no datum is shifted),
    and every ring is measured as the region its geodesic edges enclose
    on that CRS's own ellipsoid, as `distance` measures two points.
    The result is in the unit of the CRS's `x` and `y` squared
    (such as square US survey feet for EPSG:2263),
    or in square metres when those are degrees.
    That is the smaller side of the ring:
    a ring around more than half the ellipsoid is taken to enclose the rest of it.

    A geometry without a CRS is refused while the plan is built,
    as is a CRS that is not defined on longitude/latitude.

    Polygons, multipolygons, holes, 'z' and 'm', and other geometries
    (including `allow_non_polygons`) are treated as in `area_planar`.

    ```python
    df.select(geo.area_geodesic("parcel"))
    ```
    """
    return on_geometry(
        geometry, partial(_geodesic, allow_non_polygons=allow_non_polygons)
    )


def _area(
    column: pl.Expr, geometry: GeoArrowType, *, allow_non_polygons: bool
) -> pl.Expr:
    """`area_planar` where the plane keeps areas, `area_geodesic` everywhere else."""
    measure = (
        _planar
        if not geometry._declares_crs() or geometry._is_equal_area()
        else _geodesic
    )
    return measure(column, geometry, allow_non_polygons=allow_non_polygons)


def area(geometry: IntoExprColumn, *, allow_non_polygons: bool = False) -> pl.Expr:
    """The area a geometry encloses.

    Forwards to `area_planar` for a geometry without a CRS or with an equal-area CRS,
    and to `area_geodesic` for every other CRS.

    ```python
    df.select(geo.area("parcel"))
    ```
    """
    return on_geometry(geometry, partial(_area, allow_non_polygons=allow_non_polygons))
