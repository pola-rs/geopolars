"""Unit tests for `geo.snap_to_grid`."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import PointXY
from tests.unit.conftest import XY, XYM, XYZ, XYZM, Dimension, coordinates

NAN = float("nan")


def _points(dimension: Dimension, x: list[float], y: list[float]) -> pl.DataFrame:
    """Points at `x` and `y`, with a `z` and `m` that are off every grid used here."""
    rows = len(x)
    return pl.DataFrame(
        {
            "x": x,
            "y": y,
            "z": [10.3, 20.7, 0.1][:rows],
            "m": [100.3, 200.7, 300.1][:rows],
        }
    ).select(dimension.point())


def _snapped_x(x: list[float], **kwargs: object) -> list[float]:
    df = pl.DataFrame({"x": x, "y": [0.0] * len(x)}).select(XY.point())
    out = df.select(geo.snap_to_grid("point", (1.0, 1.0), **kwargs))  # type: ignore[arg-type]
    return coordinates(out)["x"].to_list()


def test_snaps_x_and_y(dimension: Dimension) -> None:
    """`z` and `m` stay as they are without a `z` spacing."""
    df = _points(dimension, [1.2, -2.6, 0.24], [3.1, 4.5, -1.1])
    out = df.select(geo.snap_to_grid("point", (0.5, 2.0)))

    assert out.schema == df.schema
    expected = coordinates(df).with_columns(
        x=pl.Series([1.0, -2.5, 0.0]), y=pl.Series([4.0, 4.0, -2.0])
    )
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize("dimension", [XYZ, XYZM], ids=["PointXYZ", "PointXYZM"])
def test_a_z_spacing_snaps_z_but_not_m(dimension: Dimension) -> None:
    df = _points(dimension, [1.2, -2.6, 0.24], [3.1, 4.5, -1.1])
    out = df.select(geo.snap_to_grid("point", (0.0, 0.0, 1.0)))

    expected = coordinates(df).with_columns(z=pl.Series([10.0, 21.0, 0.0]))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize("dimension", [XY, XYM], ids=["PointXY", "PointXYM"])
def test_a_z_spacing_needs_a_z_at_plan_time(dimension: Dimension) -> None:
    lf = _points(dimension, [1.2], [3.1]).lazy()

    with pytest.raises(TypeError, match="cannot snap z to a grid"):
        lf.select(geo.snap_to_grid("point", (1.0, 1.0, 1.0))).collect_schema()


def test_the_origin_offsets_the_grid() -> None:
    df = _points(XYZ, [1.2], [3.1])
    out = df.select(geo.snap_to_grid("point", (1.0, 1.0, 2.0), origin=(0.5, 0.25, 1.0)))

    expected = pl.DataFrame({"x": [1.5], "y": [3.25], "z": [11.0]})
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_a_spacing_of_zero_leaves_the_axis_as_it_is() -> None:
    df = _points(XY, [1.2, -2.6], [3.1, 4.6])
    out = df.select(geo.snap_to_grid("point", (0.0, 1.0)))

    expected = coordinates(df).with_columns(y=pl.Series([3.0, 5.0]))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize(
    ("mode", "snapped"),
    [
        ("half_to_even", [0.0, 0.0, 2.0, 2.0, -2.0, 1.0, -1.0]),
        ("half_away_from_zero", [1.0, -1.0, 2.0, 3.0, -3.0, 1.0, -1.0]),
        ("to_zero", [0.0, 0.0, 1.0, 2.0, -2.0, 0.0, 0.0]),
    ],
)
def test_the_mode_picks_the_grid_line(mode: str, snapped: list[float]) -> None:
    x = [0.5, -0.5, 1.5, 2.5, -2.5, 0.6, -0.6]

    assert _snapped_x(x, mode=mode) == snapped


def test_ties_to_even_count_from_the_origin() -> None:
    """From 0.5, 2.0 is one and a half spacings out: the even one is at 2.5, not 2.0."""
    assert _snapped_x([2.0], origin=(0.5, 0.0)) == [2.5]


def test_to_zero_goes_towards_the_origin() -> None:
    assert _snapped_x([2.4, -1.0], origin=(0.5, 0.0), mode="to_zero") == [1.5, -0.5]


def test_nan_the_empty_point_and_null_stay_as_they_are() -> None:
    rows = [{"x": NAN, "y": 1.2}, {"x": NAN, "y": NAN}, None]
    storage = pl.Series("point", rows, dtype=PointXY._geo_storage)
    df = pl.DataFrame(storage.ext.to(PointXY()))
    out = df.select(geo.snap_to_grid("point", (0.5, 0.5)))

    expected = [{"x": NAN, "y": 1.0}, {"x": NAN, "y": NAN}, None]
    assert_series_equal(
        out["point"].ext.storage(),
        pl.Series("point", expected, dtype=PointXY._geo_storage),
    )


def test_every_vertex_of_every_ring_snaps(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The rings keep their lengths, and stay closed."""
    off_grid = ring_coords.with_columns(pl.col("x", "y") + 0.3)
    df = dimension.polygons(off_grid)

    assert_frame_equal(
        df.select(geo.snap_to_grid("polygon", (1.0, 1.0))),
        dimension.polygons(ring_coords),
        check_exact=True,
    )


def test_every_polygon_of_a_multipolygon_snaps(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    off_grid = multipolygon_coords.with_columns(pl.col("x", "y") - 0.3)
    df = dimension.multipolygons(off_grid)

    assert_frame_equal(
        df.select(geo.snap_to_grid("multipolygon", (1.0, 1.0))),
        dimension.multipolygons(multipolygon_coords),
        check_exact=True,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"spacing": (math.nan, 1.0)}, "`spacing`"),
        ({"spacing": (1.0, math.inf)}, "`spacing`"),
        ({"spacing": (1.0, 1.0, -1.0)}, "`spacing`"),
        ({"spacing": (1.0,)}, "`spacing`"),
        ({"spacing": (1.0, 1.0, 1.0, 1.0)}, "`spacing`"),
        ({"spacing": (1.0, 1.0), "origin": (math.nan, 0.0)}, "`origin`"),
        ({"spacing": (1.0, 1.0), "mode": "half_up"}, "`mode`"),
    ],
)
def test_bad_arguments_are_refused(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        geo.snap_to_grid("point", **kwargs)  # type: ignore[arg-type]


def test_the_namespace_forwards() -> None:
    df = _points(XYZ, [1.2, -2.6, 0.24], [3.1, 4.5, -1.1])

    assert_frame_equal(
        df.select(
            gpl.col("point").geo.snap_to_grid(
                (0.5, 2.0, 3.0), (0.1, 0.2, 0.3), mode="to_zero"
            )
        ),
        df.select(
            geo.snap_to_grid(
                "point", (0.5, 2.0, 3.0), origin=(0.1, 0.2, 0.3), mode="to_zero"
            )
        ),
    )
