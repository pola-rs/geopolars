from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from tests.unit.conftest import Dimension, coordinates


def test_agg_nests_the_geometry_in_a_list(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point()).with_columns(group=pl.lit("a", pl.String))
    grouped = df.group_by("group").agg("point")

    assert grouped.schema["point"] == pl.List(dimension.point_dtype())


def test_explode_gives_the_geometry_back(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A round trip through a group:
    aggregating and exploding is the identity, dtype included."""
    df = coords.select(dimension.point()).with_columns(group=pl.lit("a", pl.String))
    out = (
        df.group_by("group")
        .agg("point")
        .explode("point", empty_as_null=False)
        .drop("group")
    )

    assert out.schema["point"] == dimension.point_dtype()
    assert_frame_equal(coordinates(out), coords.select(dimension.coords))


def test_grouping_by_a_geometry_keeps_its_dtype(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A point can be a group key, and the key column stays a point."""
    df = coords.select(dimension.point())
    grouped = df.group_by("point").len()

    assert grouped.schema["point"] == dimension.point_dtype()
    assert grouped.height == df.height
