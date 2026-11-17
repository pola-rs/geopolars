"""The default `early_coord_check` of every GEOS kernel."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError

from geopolars import geo
from geopolars.datatypes import LineStringXY

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))


def _lines(name: str, *lines: list[tuple[float, float]]) -> pl.Series:
    rows = [[{"x": float(x), "y": float(y)} for x, y in line] for line in lines]
    return pl.Series(name, rows, dtype=_XY_VERTICES).ext.to(LineStringXY())


_LINES = pl.DataFrame(
    [
        _lines("a", [(0, 0), (float("nan"), 1), (2, 2)]),
        _lines("b", [(0, 0), (4, 0)]),
    ]
)


@pytest.mark.parametrize(
    "expr",
    [
        geo.buffer("a", 1.0),
        geo.convex_hull("a"),
        geo.simplify("a", 0.1),
        geo.offset_curve("a", 1.0),
        geo.line_interpolate_point("a", 0.5),
        geo.relate("a", "b"),
        geo.is_topologically_equal("a", "b"),
        geo.shortest_line_between("a", "b"),
        geo.intersection("a", "b"),
        geo.difference("a", "b"),
    ],
    ids=[
        "buffer",
        "convex_hull",
        "simplify",
        "offset_curve",
        "line_interpolate_point",
        "relate",
        "is_topologically_equal",
        "shortest_line_between",
        "intersection",
        "difference",
    ],
)
def test_every_geos_kernel_checks_coordinates_by_default(expr: pl.Expr) -> None:
    with pytest.raises(ComputeError, match="GEOS expects finite x and y"):
        _LINES.select(expr)
