"""A `pl.Expr` that a type checker can see this package's namespaces on.

`pl.col("a").geo...` works in runtime, but type checking fails.
`pl.api.register_expr_namespace` patches a namespace onto `pl.Expr` at runtime,
which a type checker cannot follow.
`gpl.col("a")` is the same thing with the namespaces declared ahead of time.
"""

from __future__ import annotations

from typing import cast

import polars as pl

from geopolars.expr.geo import ExprGeoNameSpace


class PluginExpr(pl.Expr):
    """A `pl.Expr` that declares this package's namespaces for type checkers.
    The subclass exists so a checker can resolve `.geo`.
    """

    geo: ExprGeoNameSpace


def col(name: str) -> PluginExpr:
    """`pl.col`, typed so the plugin namespaces resolve."""
    return cast(PluginExpr, pl.col(name))


def as_plugin(expr: pl.Expr) -> PluginExpr:
    """Re-type an arbitrary expression so the plugin namespaces resolve.

    Needed when chaining through a built-in namespace, which hands back a plain
    `pl.Expr` and loses the annotation:

    .. code-block:: python

        as_plugin(pl.col("a").struct.field("b")).geo.translate((1.0, 2.0))
    """
    return cast(PluginExpr, expr)
