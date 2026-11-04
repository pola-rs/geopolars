"""The point at a distance along a line.
Deliberately not called 'interpolate', which would match geopandas etc, but clash with Polars itself.
Forwarded to GEOS (B-tier).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
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
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """The point `distance` along every line, from its start.
    With `mode="absolute"` (default), `distance` is in the units of the coordinates
    (degrees for longitude/latitude).
    With `mode="normalized"`, it is a fraction of the length of the line:
    `0.5` gives the point halfway.
    A negative `distance` counts back from the end.
    A `distance` beyond either end gives that end.
    `distance` is one number for every geometry,
    or a column (or expression) with one distance per geometry.
    A null distance gives a null point.
    A NaN or infinite distance in a column raises.
    With `ignore_per_row_errors=True`, such a row gives a null point instead.

    Only linestrings and multilinestrings are accepted.
    A multilinestring is walked part after part,
    without counting the gaps between them.
    Empty parts are skipped.
    The point keeps the dimension and CRS of the line:
    computed by GEOS, which interpolates z and m too.
    {empty_and_null}

    {geos_errors}

    {geos_non_finite}
    {early_finite_check}

    ```python
    df.select(geo.line_interpolate_point("road", 100.0))
    df.select(geo.line_interpolate_point("road", 0.5, mode="normalized"))
    ```
    """
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
                "ignore_per_row_errors": ignore_per_row_errors,
                "early_finite_check": early_finite_check,
            },
            is_elementwise=True,
        )

    return on_geometry(geometry, build)
