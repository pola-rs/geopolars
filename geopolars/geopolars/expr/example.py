"""Namespaces of the pyo3-polars example plugin this repo was forked from.

Kept as worked examples; new geometry work goes in `geopolars.expr.geometry`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars import date_util, dist, language, panic

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn


@pl.api.register_expr_namespace("language")
class Language:
    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def pig_latinnify(self, capitalize: bool = False) -> pl.Expr:
        return language.pig_latinnify(self._expr, capitalize=capitalize)

    def append_args(
        self,
        float_arg: float,
        integer_arg: int,
        string_arg: str,
        boolean_arg: bool,
    ) -> pl.Expr:
        return language.append_args(
            self._expr,
            float_arg=float_arg,
            integer_arg=integer_arg,
            string_arg=string_arg,
            boolean_arg=boolean_arg,
        )


@pl.api.register_expr_namespace("dist")
class Distance:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def hamming_distance(self, other: IntoExprColumn) -> pl.Expr:
        return dist.hamming_distance(self._expr, other)

    def jaccard_similarity(self, other: IntoExprColumn) -> pl.Expr:
        return dist.jaccard_similarity(self._expr, other)

    def haversine(
        self,
        start_long: IntoExprColumn,
        end_lat: IntoExprColumn,
        end_long: IntoExprColumn,
    ) -> pl.Expr:
        return dist.haversine(self._expr, start_long, end_lat, end_long)


@pl.api.register_expr_namespace("date_util")
class DateUtil:
    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def is_leap_year(self) -> pl.Expr:
        return date_util.is_leap_year(self._expr)

    def change_time_zone(self, tz: str = "Europe/Amsterdam") -> pl.Expr:
        return date_util.change_time_zone(self._expr, tz=tz)


@pl.api.register_expr_namespace("panic")
class Panic:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def panic(self) -> pl.Expr:
        return panic.panic(self._expr)
