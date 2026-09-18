"""The `geometry` expression namespace."""

from __future__ import annotations

import polars as pl

from geopolars.geometry import affine, construct


@pl.api.register_expr_namespace("geometry")
class Geometry:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def linestring(self) -> pl.Expr:
        return construct.linestring_from_vertices(self._expr)

    def polygon(self) -> pl.Expr:
        return construct.polygon_from_rings(self._expr)

    def translate(self, dx: float, dy: float, dz: float = 0.0) -> pl.Expr:
        return affine.translate(self._expr, dx=dx, dy=dy, dz=dz)
