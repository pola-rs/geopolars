"""Building `geoarrow.point` columns with `geopolars.geometry.point`."""

from __future__ import annotations

import polars as pl
import pytest
from geopolars.datatypes import GeoPoint, PointXY, PointXYM, PointXYZ, PointXYZM
from polars.testing import assert_frame_equal

from geopolars import geometry
from tests.unit.conftest import coordinates


@pytest.mark.parametrize(
    ("optional", "expected"),
    [
        ((), PointXY),
        (("z",), PointXYZ),
        (("m",), PointXYM),
        (("z", "m"), PointXYZM),
    ],
    ids=["xy", "xyz", "xym", "xyzm"],
)
def test_optional_coordinates_pick_the_dtype(
    coords: pl.DataFrame, optional: tuple[str, ...], expected: type[GeoPoint]
) -> None:
    """Which of `z`/`m` you pass is the only thing that sets the dimension."""
    optional_coords = {name: name for name in optional}
    df = coords.select(geometry.point("x", "y", **optional_coords).alias("point"))

    assert df.schema["point"] == expected()


def test_coordinates_are_stored_in_spec_order(coords: pl.DataFrame) -> None:
    """`m` is the last argument but not always the last field:
    the spec fixes x, y, z, m, so an XYM point stores m third."""
    df = coords.select(geometry.point("x", "y", m="m").alias("point"))

    assert_frame_equal(coordinates(df), coords.select("x", "y", "m"))


@pytest.mark.parametrize("form", ["column name", "expression", "series"])
def test_accepts_every_into_expr_column_form(coords: pl.DataFrame, form: str) -> None:
    """Coordinates may be given as a column name, an expression, or a Series."""
    if form == "column name":
        x, y = "x", "y"
    elif form == "expression":
        x, y = pl.col("x"), pl.col("y")
    else:
        x, y = coords["x"], coords["y"]

    df = coords.select(geometry.point(x, y).alias("point"))

    assert df.schema["point"] == PointXY()
    assert_frame_equal(coordinates(df), coords.select("x", "y"))


def test_coordinates_are_computable() -> None:
    """An expression argument is evaluated, not just referenced by name."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(
        geometry.point(pl.col("x") * 10, pl.col("y") + 1).alias("point")
    )

    assert_frame_equal(coordinates(df), pl.DataFrame({"x": [10.0], "y": [3.0]}))


def test_integer_coordinates_are_accepted() -> None:
    """Storage is f64 by definition, so integer degrees are widened rather
    than rejected."""
    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]}).select(
        geometry.point("x", "y").alias("point")
    )

    assert df.schema["point"] == PointXY()
    assert_frame_equal(
        coordinates(df), pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    )


def test_a_missing_coordinate_invalidates_the_whole_point() -> None:
    """Missing coordinate must invalidate the entire point.
    See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {
            "x": [None, 1.0, 1.0, 1.0],
            "y": [2.0, None, 2.0, 2.0],
            "z": [3.0, 3.0, None, 3.0],
        }
    ).select(geometry.point("x", "y", z="z").alias("point"))

    assert df["point"].is_null().to_list() == [True, True, True, False]
    assert_frame_equal(
        coordinates(df),
        pl.DataFrame(
            {
                "x": [None, None, None, 1.0],
                "y": [None, None, None, 2.0],
                "z": [None, None, None, 3.0],
            }
        ),
    )


def test_empty_frame_keeps_its_dtype() -> None:
    """A zero-row build still produces a point column, not a bare struct."""
    df = pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64}).select(
        geometry.point("x", "y").alias("point")
    )

    assert df.height == 0
    assert df.schema["point"] == PointXY()
