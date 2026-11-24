"""Unit tests for `geo.rotate_about_centroid`."""

from __future__ import annotations

import math
from typing import Literal

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from tests.unit.conftest import XY, XYM, XYZ, XYZM, Dimension, coordinates

RD = "EPSG:28992"


def test_a_polygon_turns_about_its_centroid(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords)

    assert_frame_equal(
        df.select(geo.rotate_about_centroid("polygon", 30)),
        df.select(geo.rotate("polygon", 30, origin=geo.centroid("polygon"))),
        check_exact=True,
    )


@pytest.mark.parametrize(
    ("dimension", "axis"),
    [(XY, "z"), (XYM, "z"), (XYZ, "z"), (XYZM, "z"), (XYZ, "x"), (XYZM, "y")],
    ids=["XY", "XYM", "XYZ", "XYZM", "XYZ about x", "XYZM about y"],
)
def test_a_line_turns_about_its_centroid(
    line_coords: pl.DataFrame, dimension: Dimension, axis: Literal["x", "y", "z"]
) -> None:
    """A line's centroid keeps its `z`, so it can turn out of the plane too."""
    df = dimension.lines(line_coords)

    assert_frame_equal(
        df.select(geo.rotate_about_centroid("line", 30, axis=axis)),
        df.select(geo.rotate("line", 30, axis=axis, origin=geo.centroid("line"))),
        check_exact=True,
    )


def test_the_centroid_stays_where_it_is(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords)
    turned = df.select(geo.rotate_about_centroid("polygon", math.pi / 4, unit="rad"))

    assert_frame_equal(
        coordinates(turned.select(geo.centroid("polygon")), "polygon"),
        coordinates(df.select(geo.centroid("polygon")), "polygon"),
    )


def test_each_geometry_turns_about_its_own_centroid(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords)
    out = df.select(geo.rotate_about_centroid("polygon", 30))

    for row in range(df.height):
        alone = df.slice(row, 1).select(geo.rotate_about_centroid("polygon", 30))
        assert_frame_equal(out.slice(row, 1), alone, check_exact=True)


@pytest.mark.parametrize("dimension", [XYZ, XYZM], ids=["XYZ", "XYZM"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_a_polygon_turns_out_of_the_plane_about_its_centroid(
    ring_coords: pl.DataFrame, dimension: Dimension, axis: Literal["x", "y"]
) -> None:
    """A polygon's centroid keeps its `z`, so the axis goes through its height."""
    df = dimension.polygons(ring_coords)

    assert_frame_equal(
        df.select(geo.rotate_about_centroid("polygon", 30, axis=axis)),
        df.select(geo.rotate("polygon", 30, axis=axis, origin=geo.centroid("polygon"))),
        check_exact=True,
    )


def test_only_lines_and_polygons_are_accepted(
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> None:
    for df in [
        coords.select(XY.point()),
        XY.multipoints(line_coords),
        XY.multilinestrings(ring_coords),
        XY.multipolygons(multipolygon_coords),
    ]:
        (name,) = df.columns
        with pytest.raises(TypeError, match="cannot rotate about its centroid"):
            df.lazy().select(geo.rotate_about_centroid(name, 30)).collect_schema()


def test_bad_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="`axis`"):
        geo.rotate_about_centroid("line", 30, axis="w")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`unit`"):
        geo.rotate_about_centroid("line", 30, unit="pi")  # type: ignore[arg-type]


def test_a_null_comes_out_null_and_an_empty_geometry_stays_empty() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 0.0}], [], None]},
        schema={"line": pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))},
    ).select(pl.col("line").ext.to(XY.linestring_dtype()))
    out = df.select(geo.rotate_about_centroid("line", 90))

    assert out["line"].ext.storage()[1:].to_list() == [[], None]
    assert out["line"].ext.storage()[0].to_list() == [
        {"x": 1.0, "y": -1.0},
        {"x": 1.0, "y": 1.0},
    ]


def test_a_line_without_length_goes_to_on_row_error() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.0}]]},
        schema={"line": pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))},
    ).select(pl.col("line").ext.to(XY.linestring_dtype()))

    with pytest.raises(pl.exceptions.ComputeError, match="got one without"):
        df.select(geo.rotate_about_centroid("line", 90))
    out = df.select(geo.rotate_about_centroid("line", 90, on_row_error="null"))

    assert out["line"].is_null().to_list() == [True]


def test_the_crs_is_carried_over(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords).select(
        pl.col("polygon").ext.storage().ext.to(XY.polygon_dtype(crs=RD))
    )
    out = df.lazy().select(geo.rotate_about_centroid("polygon", 30))

    assert out.collect_schema()["polygon"] == df.schema["polygon"]
    out.collect()


def test_the_namespace_forwards(line_coords: pl.DataFrame) -> None:
    df = XYZ.lines(line_coords)

    assert_frame_equal(
        df.select(gpl.col("line").geo.rotate_about_centroid(math.pi / 4, "rad", "x")),
        df.select(geo.rotate_about_centroid("line", 45, axis="x")),
    )
