"""A `pl.Expr` that a type checker can see this package's namespaces on.

`pl.api.register_expr_namespace` patches a namespace onto `pl.Expr` at runtime,
which a type checker cannot follow. `pl.col("a").geometry...` works regardless;
`gpl.col("a")` is the same thing with the namespaces declared ahead of time.
"""

from __future__ import annotations

from typing import cast

import polars as pl

from geopolars.expr.example import DateUtil, Distance, Language, Panic
from geopolars.expr.geometry import Geometry


class PluginExpr(pl.Expr):
    """A `pl.Expr` that declares this package's namespaces for type checkers.
    The subclass exists so a checker can resolve `.geometry`, `.dist`, etc.
    """

    geometry: Geometry
    language: Language
    dist: Distance
    date_util: DateUtil
    panic: Panic


def col(name: str) -> PluginExpr:
    """`pl.col`, typed so the plugin namespaces resolve."""
    return cast(PluginExpr, pl.col(name))


def as_plugin(expr: pl.Expr) -> PluginExpr:
    """Re-type an arbitrary expression so the plugin namespaces resolve.

    Needed when chaining through a built-in namespace, which hands back a plain
    `pl.Expr` and loses the annotation:

    ```python
    as_plugin(pl.col("a").str.to_uppercase()).dist.hamming_distance("b")
    ```
    """
    return cast(PluginExpr, expr)
