"""Review reproducers for the hausdorff branch (git-excluded)."""

from __future__ import annotations

import math

import polars as pl

from geopolars import geo
from geopolars.datatypes import (
    PolygonXY,
)
from tests.unit.operations.test_hausdorff_line_between import (
    _as_wkt,
    _line_strings,
    _wkt,
)

# Behaviour: empty against failing (sibling shortest_line_between gives LINESTRING EMPTY).


def test_empty_against_failing_is_empty_like_shortest_line() -> None:
    df = pl.DataFrame(
        [
            _wkt("a", PolygonXY(), ["POLYGON EMPTY"]),
            _line_strings("b", [[(0, 0), (math.nan, 1)]]),
        ]
    )
    out = df.select(geo.hausdorff_line_between("a", "b", ignore_per_row_errors=True))
    assert _as_wkt(out["a"]) == ["LINESTRING EMPTY"]


def test_empty_against_failing_does_not_raise() -> None:
    df = pl.DataFrame(
        [
            _wkt("a", PolygonXY(), ["POLYGON EMPTY"]),
            _line_strings("b", [[(0, 0), (math.nan, 1)]]),
        ]
    )
    out = df.select(geo.hausdorff_line_between("a", "b"))
    assert _as_wkt(out["a"]) == ["LINESTRING EMPTY"]
