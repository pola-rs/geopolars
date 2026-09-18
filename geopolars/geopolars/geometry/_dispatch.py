"""Choosing an expression by the dtype of the column it is given.

We need to be able to predict what a schema could look like.
Normally this is the responsibility of Polars itself,
but with our struct types and the FFI boundary crossing,
it can use every bit of help it can get.

We do this using a *selector*:
An expression built on a selector vanishes unless the column
is the geometry it was written for,
and `coalesce` is what gathers the one that is left back into a single expression.
This only works on a *column*, whose dtype the schema knows.

What may be built on a vanishing expression is not everything:
methods and operators on it go along quietly, and so does `pl.when`
anything variadic (e.g. `pl.struct`, `pl.all_horizontal`, `pl.concat_list`)
sees no inputs left and either raises or collapses to a literal,
so a vanishing expression cannot be handed to one.
`coalesce` first, then build.
`pl.coalesce` itself needs an input that is still there,
which a literal gives it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from geopolars.datatypes import GEOMETRIES, GeoArrowType

if TYPE_CHECKING:
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

# The column, where its dtype is one of these geometries; `None` if it cannot be.
Where = Callable[..., "pl.Expr | None"]

# The branches to choose between, one per geometry the column could hold.
Build = Callable[[Where], Iterable[pl.Expr]]


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


def _unsupported(name: str) -> pl.Expr:
    """The branch left standing when the column is no geometry of ours.
    this is the 'error' case."""
    rest = cs.by_name(name) - cs.by_dtype(*(g() for g in GEOMETRY_TYPES))
    return rest.struct.field(UNSUPPORTED)


def by_geometry(value: IntoExprColumn, build: Build) -> pl.Expr:
    """Build an expression for whichever geometry `value` turns out to hold.
    Tries out all the different types, and returns if one fits the criteria.
    """
    if isinstance(value, pl.Series):
        # A Series carries its dtype with it, so there is nothing to choose.
        geometry = _geometry_of(value.dtype)
        column = pl.lit(value)

        def known(*types: type[GeoArrowType]) -> pl.Expr | None:
            return column if geometry in types else None

        # The branches are named after whatever they were built from, so the
        # column's own name has to be put back on the one that survived.
        return pl.coalesce(list(build(known))).alias(value.name)

    name = _column(value)

    def selecting(*types: type[GeoArrowType]) -> pl.Expr | None:
        return cs.by_name(name) & cs.by_dtype(*(geometry() for geometry in types))

    return pl.coalesce([*build(selecting), _unsupported(name)]).alias(name)
