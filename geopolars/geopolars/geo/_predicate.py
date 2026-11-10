"""The shared plumbing of the predicates on two geometries, forwarded to GEOS."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.geo._dispatch import ensure_same_crs, on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def predicate(
    a: IntoExprColumn,
    b: IntoExprColumn,
    function_name: str,
    *,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> pl.Expr:
    kwargs = {
        "ignore_per_row_errors": ignore_per_row_errors,
        "early_finite_check": early_finite_check,
    }

    def build(
        a: pl.Expr, a_dtype: GeoArrowType, b: pl.Expr, b_dtype: GeoArrowType
    ) -> pl.Expr:
        ensure_same_crs(a_dtype, b_dtype)
        return register_plugin_function(
            plugin_path=LIB,
            args=[a, b],
            function_name=function_name,
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
