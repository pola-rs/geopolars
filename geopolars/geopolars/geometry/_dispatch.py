"""Choosing an expression by the dtype of the column it is given.

An expression is spelled out before Polars resolves a schema,
so Python cannot look at a column's dtype while building one.
We still need this functionality.

We do this using a *selector*:
this only works on a *column*, whose dtype the schema knows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from geopolars.datatypes import GEOMETRIES, GeoArrowType

if TYPE_CHECKING:
    from collections.abc import Iterator

    from geopolars._typing import IntoExprColumn

# analog of `Kind::ALL` x `Dimension::ALL` on the Rust side.
GEOMETRY_TYPES: tuple[type[GeoArrowType], ...] = tuple(
    concrete for geometry in GEOMETRIES for concrete in geometry.dimensions()
)


def _names() -> str:
    """The extension names, for an error message. Mirrors `Kind::names`."""
    names = [f"`{geometry._extension_name}`" for geometry in GEOMETRIES]
    return f"{', '.join(names[:-1])} or {names[-1]}"


UNSUPPORTED = (
    f"expected a {_names()} column with separated x/y[/z][/m] coordinates "
    "and no extension metadata, which this version cannot carry through"
)

# What a branch is built from, once the dtype is known.
# the geometry the column holds, and the column itself.
Build = Callable[[type[GeoArrowType], pl.Expr], pl.Expr]


def _column(value: IntoExprColumn) -> str:
    """The name of the column an expression is to be built for."""
    if isinstance(value, str):
        return value
    if isinstance(value, pl.Expr) and value.meta.is_column():
        return value.meta.output_name()

    msg = f"expected a column name or a `pl.col(...)`. \
        this operation is chosen by the geometry's dtype, \
        which Polars resolves for a column and not for an expression. \
        Select the expression into a column first. \
        got: {value!r}"
    raise TypeError(msg)


def _geometry_of(dtype: pl.DataType) -> type[GeoArrowType]:
    geometry = type(dtype)
    if geometry not in GEOMETRY_TYPES or dtype.ext_metadata() is not None:
        msg = f"{UNSUPPORTED}, got: {dtype!r}"
        raise TypeError(msg)
    return geometry


def _branches(name: str, build: Build) -> Iterator[pl.Expr]:
    """One branch per geometry dtype, each selecting only the column it fits."""
    for geometry in GEOMETRY_TYPES:
        yield build(geometry, cs.by_name(name) & cs.by_dtype(geometry()))

    # All of the errors raise an error if it fails.
    # If one of them does not raise an error, it must be that type.
    yield (cs.by_name(name) - cs.by_dtype(*(g() for g in GEOMETRY_TYPES))).struct.field(
        UNSUPPORTED
    )


def by_geometry(value: IntoExprColumn, build: Build) -> pl.Expr:
    """Build an expression for whichever geometry `value` turns out to hold."""
    if isinstance(value, pl.Series):
        # A Series carries its dtype with it, so there is nothing to choose.
        return build(_geometry_of(value.dtype), pl.lit(value))

    return pl.coalesce(list(_branches(_column(value), build)))
