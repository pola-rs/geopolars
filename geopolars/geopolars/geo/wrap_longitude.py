"""Bringing longitudes that went past the antimeridian back into range."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _turns(x: pl.Expr, start: float, turn: float) -> pl.Expr:
    """How many whole turns `x` lies outside `[start, start + turn]`;
    0 inside it (or NaN)."""
    end = start + turn
    return (
        pl.when(x > end)
        .then(((x - end) / turn).ceil())
        .when(x < start)
        .then(((x - start) / turn).floor())
        .otherwise(0.0)
    )


def _shifts(values: pl.Expr, layers: int, start: float, turn: float) -> pl.Expr:
    """`_turns` for every coordinate, keeping the geometry's nesting."""
    if layers == 0:
        return _turns(values.struct.field("x"), start, turn)
    return values.list.eval(_shifts(pl.element(), layers - 1, start, turn))


def _wrapped(values: pl.Expr, layers: int, start: float, turn: float) -> pl.Expr:
    """Every coordinate with its `x` moved back by its own whole turns.
    The other coordinates and the list offsets are left as they are."""
    if layers == 0:
        x = pl.field("x")
        return values.struct.with_fields(x=x - _turns(x, start, turn) * turn)
    return values.list.eval(_wrapped(pl.element(), layers - 1, start, turn))


def _crossing(storage: pl.Expr, layers: int, start: float, turn: float) -> pl.Expr:
    """Whether a geometry's coordinates would not all move by the same turns."""
    shifts = _shifts(storage, layers, start, turn)
    # One flat list of shifts per geometry, over every part and ring.
    for _ in range(layers - 1):
        shifts = shifts.list.eval(pl.element().explode())
    return shifts.list.min() != shifts.list.max()


def _wrap(
    start: float | None, skip_crossing: bool
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        turn = geometry._longitude_turn()
        if turn is None:
            msg = (
                f"{geometry!r} has no longitude to wrap: "
                "`wrap_longitude` needs a geographic CRS, such as EPSG:4326"
            )
            raise TypeError(msg)
        # Centred on the prime meridian unless told otherwise.
        lowest = -turn / 2 if start is None else start

        storage = column.ext.storage()
        layers = geometry._nesting
        wrapped = _wrapped(storage, layers, lowest, turn)
        if skip_crossing and layers > 0:
            crossing = _crossing(storage, layers, lowest, turn)
            wrapped = pl.when(crossing).then(storage).otherwise(wrapped)
        # The input's own dtype: the coordinates moved, the CRS did not.
        return wrapped.ext.to(geometry)

    return build


def wrap_longitude(
    geometry: IntoExprColumn,
    *,
    start: float | None = None,
    skip_crossing: bool = False,
) -> pl.Expr:
    """Move every `x` that went past the edge of its range back into it by whole turns.

    The range is one full turn in the CRS's unit (360 for degrees, 400 for grads),
    from `start` to `start + turn`.
    By default it is centred on the prime meridian: `[-180, 180]` for degrees,
    so 190 becomes -170, -200 becomes 160, and 540 becomes 180.
    Data that runs from 0 to 360 instead, as many climate and ocean grids do,
    passes `start=0`: then -10 becomes 350, and 370 becomes 10.
    The CRS cannot say which convention data uses (EPSG:4326 covers both),
    which is why it is up to the caller.
    Coordinates already in range are left as they are, including both ends.
    Only `x` moves: `y`, `z` and `m` are untouched.

    Each vertex moves on its own, which can tear a geometry apart.
    A line from 170 to 190 becomes one from 170 to -170, which runs the long way round.
    With `skip_crossing`, a geometry whose vertices would not all move by the same number
    of turns is left as it is, still past the edge.
    A geometry that lies entirely past it is still moved, and moves as a whole.

    The geometry needs a geographic CRS: without one there is no longitude to wrap,
    and the expression is refused while the plan is built.

    .. code-block:: python

        df.select(geo.wrap_longitude("route", skip_crossing=True))
    """
    if start is not None and not math.isfinite(start):
        msg = f"`start` has to be a finite number, got: {start!r}"
        raise ValueError(msg)
    return on_geometry(geometry, _wrap(start, skip_crossing))
