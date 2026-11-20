"""Unit tests for `geo.skew_about_centroid`."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from tests.unit.conftest import XY, XYZ, Dimension, coordinates

RD = "EPSG:28992"


def test_a_polygon_skews_about_its_centroid(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords)

    assert_frame_equal(
        df.select(geo.skew_about_centroid("polygon", 30, -15)),
        df.select(geo.skew("polygon", 30, -15, origin=geo.centroid("polygon"))),
        check_exact=True,
    )


def test_a_line_skews_about_its_centroid(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.lines(line_coords)

    assert_frame_equal(
        df.select(geo.skew_about_centroid("line", 30, -15)),
        df.select(geo.skew("line", 30, -15, origin=geo.centroid("line"))),
        check_exact=True,
    )


def test_a_polygon_with_z_can_skew(ring_coords: pl.DataFrame) -> None:
    """Skewing stays in the plane, so the centroid having no `z` does not matter."""
    df = XYZ.polygons(ring_coords)
    out = df.select(geo.skew_about_centroid("polygon", 30))

    assert out.schema["polygon"] == df.schema["polygon"]


def test_the_centroid_stays_where_it_is(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords)
    skewed = df.select(geo.skew_about_centroid("polygon", 30, -15))

    assert_frame_equal(
        coordinates(skewed.select(geo.centroid("polygon")), "polygon"),
        coordinates(df.select(geo.centroid("polygon")), "polygon"),
    )


def test_each_geometry_skews_about_its_own_centroid(
    ring_coords: pl.DataFrame,
) -> None:
    df = XY.polygons(ring_coords)
    out = df.select(geo.skew_about_centroid("polygon", 30))

    for row in range(df.height):
        alone = df.slice(row, 1).select(geo.skew_about_centroid("polygon", 30))
        assert_frame_equal(out.slice(row, 1), alone, check_exact=True)


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
        with pytest.raises(TypeError, match="cannot skew about its centroid"):
            df.lazy().select(geo.skew_about_centroid(name, 30)).collect_schema()


def test_bad_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="quarter turn"):
        geo.skew_about_centroid("line", 90)
    with pytest.raises(ValueError, match="`unit`"):
        geo.skew_about_centroid("line", 30, unit="pi")  # type: ignore[arg-type]


def test_a_geometry_without_a_centroid_comes_out_null() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 2.0}], [], None]},
        schema={"line": pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))},
    ).select(pl.col("line").ext.to(XY.linestring_dtype()))
    out = df.select(geo.skew_about_centroid("line", 45))

    assert out["line"].is_null().to_list() == [False, True, True]
    assert out["line"].ext.storage()[0].to_list() == [
        {"x": -1.0, "y": 0.0},
        {"x": 1.0, "y": 2.0},
    ]


def test_the_crs_is_carried_over(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords).select(
        pl.col("polygon").ext.storage().ext.to(XY.polygon_dtype(crs=RD))
    )
    out = df.lazy().select(geo.skew_about_centroid("polygon", 30))

    assert out.collect_schema()["polygon"] == df.schema["polygon"]
    out.collect()


def test_the_namespace_forwards(line_coords: pl.DataFrame) -> None:
    df = XY.lines(line_coords)

    assert_frame_equal(
        df.select(
            gpl.col("line").geo.skew_about_centroid(math.pi / 8, -math.pi / 4, "rad")
        ),
        df.select(geo.skew_about_centroid("line", 22.5, -45)),
    )
