"""How long a linestring is, on the plane or along the ellipsoid of its CRS.

Which of the two is picked by default
is described under "Geographic and projected" in `docs/source/user-guide/geo/concepts/crs.md`.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import LineStringType, MultiLineStringType
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.distance import _auto_distance_method

if TYPE_CHECKING:
    from geopolars._typing import DistanceMethod, IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _has_no_length(
    column: pl.Expr, geometry: GeoArrowType, *, allow_non_lines: bool
) -> pl.Expr | None:
    """Only linestrings and multilinestrings have a length: `None` for those.
    Other geometries are refused, or given `0.0` with `allow_non_lines`."""
    if isinstance(geometry, (LineStringType, MultiLineStringType)):
        return None
    if not allow_non_lines:
        msg = (
            "a length is measured on a `geoarrow.linestring` "
            f"or `geoarrow.multilinestring` column, got: {geometry!r};"
            " pass `allow_non_lines=True` to give it a length of 0.0"
        )
        raise TypeError(msg)
    return pl.when(column.is_not_null()).then(pl.lit(0.0, dtype=pl.Float64))


def _length(
    column: pl.Expr,
    geometry: GeoArrowType,
    *,
    distance_method: DistanceMethod,
    allow_non_lines: bool,
    ignore_nan: bool,
) -> pl.Expr:
    if distance_method == "auto":
        distance_method = _auto_distance_method(geometry)
    elif distance_method == "geodesic" and not geometry._declares_crs():
        msg = "a geodesic length needs a CRS to measure on, but the geometry declares none"
        raise TypeError(msg)
    zero = _has_no_length(column, geometry, allow_non_lines=allow_non_lines)
    if zero is not None:
        return zero
    # Whether PROJ can use the CRS is for the kernel to check.
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="length",
        kwargs={"distance_method": distance_method, "ignore_nan": ignore_nan},
        is_elementwise=True,
    )


def length(
    geometry: IntoExprColumn,
    *,
    distance_method: DistanceMethod = "auto",
    allow_non_lines: bool = False,
    ignore_nan: bool = False,
) -> pl.Expr:
    """How long a linestring is, as an `f64`.
    For details on `distance_method`, see the description in `distance`.

    A multilinestring gets the sum of the lengths of its parts.
    For a line with a `z`, every segment counts its difference in height too
    (with `"geodesic"`, taken to be in the unit of the result).
    `m` is ignored.

    A vertex with a NaN in `x` or `y` (or `z`, for a line with a `z`) makes the length NaN,
    unless `ignore_nan` is set: then it is skipped, and the line runs on to the next vertex.

    An empty linestring, or one of a single vertex, has a length of `0.0`.
    A null geometry has no length.
    Every other geometry (points, polygons and their multi forms)
    is refused while the plan is built,
    unless `allow_non_lines` is set: then each of them gets `0.0`
    (and a null stays null).

    =======================  ============
    in                       out
    =======================  ============
    `PointXY[ZM]`            refused
    `LineStringXY[ZM]`       `0.0` and up
    `PolygonXY[ZM]`          refused
    `MultiPointXY[ZM]`       refused
    `MultiLineStringXY[ZM]`  `0.0` and up
    `MultiPolygonXY[ZM]`     refused
    =======================  ============

    .. code-block:: python

        df.select(geo.length("route"))
        df.select(geo.length("route", distance_method="planar"))
    """
    return on_geometry(
        geometry,
        partial(
            _length,
            distance_method=distance_method,
            allow_non_lines=allow_non_lines,
            ignore_nan=ignore_nan,
        ),
    )
