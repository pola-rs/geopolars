"""Sorting geometry columns with ordinary Polars operations."""

from __future__ import annotations

import polars as pl

from geopolars import geometry
from tests.unit.conftest import Dimension


def test_sort_keeps_the_dtype(coords: pl.DataFrame, dimension: Dimension) -> None:
    df = coords.select(dimension.point())

    assert df.sort("point").schema["point"] == dimension.point_dtype()


def test_sort_orders_by_the_coordinates() -> None:
    """Sorting a geometry sorts on the coordinates behind it,
    in spec order, so a column that only varies in x sorts by x."""
    points = pl.DataFrame(
        {"i": [2, 1, 3], "x": [2.0, 1.0, 3.0], "y": [0.0, 0.0, 0.0]}
    ).select("i", geometry.point("x", "y").alias("point"))

    assert points.sort("point")["i"].to_list() == [1, 2, 3]
    assert points.sort("point", descending=True)["i"].to_list() == [3, 2, 1]


def test_sort_keeps_a_linestrings_dtype(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.lines(line_coords)

    assert df.sort("line").schema["line"] == dimension.linestring_dtype()


def test_sort_keeps_a_polygons_dtype(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords)

    assert df.sort("polygon").schema["polygon"] == dimension.polygon_dtype()
