"""The `geometry` expression namespace."""

from __future__ import annotations

import polars as pl

from geopolars.geometry import affine, centroid, construct


@pl.api.register_expr_namespace("geometry")
class Geometry:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def linestring(self) -> pl.Expr:
        return construct.linestring_from_vertices(self._expr)

    def multipoint(self) -> pl.Expr:
        return construct.multipoint_from_points(self._expr)

    def polygon(self) -> pl.Expr:
        return construct.polygon_from_rings(self._expr)

    def validate(self) -> pl.Expr:
        return construct.validate(self._expr)

    def translate(self, dx: float, dy: float, dz: float = 0.0) -> pl.Expr:
        return affine.translate(self._expr, dx=dx, dy=dy, dz=dz)

    def coordinate_centroid(self) -> pl.Expr:
        return centroid.coordinate_centroid(self._expr)
