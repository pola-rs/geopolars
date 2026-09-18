"""Moving a geometry through space."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn


def translate(expr: IntoExprColumn, dx: float, dy: float, dz: float = 0.0) -> pl.Expr:
    """Shift every coordinate of a geometry by a constant offset.

    Works on any geometry, of any dimension. A non-zero `dz` is rejected for a
    geometry with no `z` rather than silently ignored. An `m` value is a measure,
    not a position, so it is always carried through untouched.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        function_name="translate",
        is_elementwise=True,
        kwargs={"dx": dx, "dy": dy, "dz": dz},
    )
