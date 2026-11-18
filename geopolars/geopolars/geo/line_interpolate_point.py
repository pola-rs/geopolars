"""The point at a distance along a line.
Deliberately not called 'interpolate', which would match geopandas etc, but clash with Polars itself.
Forwarded to GEOS (B-tier).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._typing import EarlyCoordCheck, OnRowError
from geopolars._utils import LIB, validate_one_of
from geopolars.datatypes import LineStringType, MultiLineStringType
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.buffer import _check_one_of, _distances_of

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

_MODES = ("absolute", "normalized")


@doc(output="point")
def line_interpolate_point(
    geometry: IntoExprColumn,
    distance: float | IntoExprColumn,
    *,
    mode: Literal["absolute", "normalized"] = "absolute",
    on_row_error: OnRowError = "raise",
    early_coord_check: EarlyCoordCheck = "finite_and_empty",
) -> pl.Expr:
    """
    Creates a point at `distance` along every line, from its start.
    With `mode="absolute"` (default), `distance` is in the units of the coordinates
    (degrees for longitude/latitude).
    With `mode="normalized"`, it is a fraction of the length of the line:
    `0.5` gives the point halfway.
    A negative `distance` counts back from the end.
    A `distance` beyond either end gives that end.
    `distance` is one number for every geometry,
    or a column (or expression) with one distance per geometry.
    A null distance gives a null point, also for an empty geometry.
    A NaN or infinite distance in a column raises.
    A linestring without vertices or a multilinestring without parts
    still gives an empty point for such a distance.

    Only linestrings and multilinestrings are accepted.
    A multilinestring is walked part after part,
    without counting the gaps between them.
    Empty parts are skipped.
    The point keeps the dimension and CRS of the line:
    computed by GEOS, which interpolates z and m too.
    {empty_and_null}

    {geos_errors}
    {on_row_error}

    {geos_non_finite}
    GEOS then does not raise, but gives a wrong point for most distances:
    often the end of the line, or a point with a NaN or infinite coordinate.
    {early_coord_check}

    .. code-block:: python

        df.select(geo.line_interpolate_point("road", 100.0))
        df.select(geo.line_interpolate_point("road", 0.5, mode="normalized"))
    """
    validate_one_of("on_row_error", on_row_error, OnRowError)
    validate_one_of("early_coord_check", early_coord_check, EarlyCoordCheck)
    constant = isinstance(distance, int | float)
    if constant and not math.isfinite(distance):
        msg = f"line_interpolate_point expects a finite distance, got: {distance!r}"
        raise ValueError(msg)
    _check_one_of("line_interpolate_point", "mode", mode, _MODES)

    def build(column: pl.Expr, dtype: GeoArrowType) -> pl.Expr:
        if not isinstance(dtype, LineStringType | MultiLineStringType):
            msg = (
                "line_interpolate_point expects a linestring or multilinestring column,"
                f" got: {dtype!r}"
            )
            raise TypeError(msg)
        args = [column] if constant else [column, _distances_of(column, distance)]
        return register_plugin_function(
            plugin_path=LIB,
            args=args,
            function_name="line_interpolate_point",
            kwargs={
                # `None` when the distances are the second argument.
                "distance": float(distance) if constant else None,
                "mode": mode,
                "on_row_error": on_row_error,
                "early_coord_check": early_coord_check,
            },
            is_elementwise=True,
        )

    return on_geometry(geometry, build)
