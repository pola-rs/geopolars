"""Moving longitudes that went past the antimeridian back into range."""

from __future__ import annotations

import math

import polars as pl
import pytest

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import LineStringXY, PointXYZM

WGS84 = "EPSG:4326"


def _x(df: pl.LazyFrame, **kwargs: float | bool) -> list:
    """The `x` coordinates of `wrap_longitude` over the frame's one column."""
    (name,) = df.collect_schema().names()
    out = df.select(geo.wrap_longitude(name, **kwargs)).select(geo.x(name))
    return out.collect().to_series().to_list()


def _points(*x: float, crs: str | None = WGS84) -> pl.LazyFrame:
    return pl.LazyFrame({"x": list(x), "y": [0.0] * len(x)}).select(
        geo.point("x", "y", crs=crs).alias("point")
    )


def _lines(*lines: list[float], crs: str = WGS84) -> pl.LazyFrame:
    return pl.LazyFrame(
        {"x": list(lines), "y": [[0.0] * len(line) for line in lines]}
    ).select(geo.line_string_from_columns("x", "y", crs=crs).alias("line"))


def test_a_point_wraps_by_whole_turns() -> None:
    assert _x(_points(190.0, -200.0, 540.0, -540.0)) == [-170.0, 160.0, 180.0, -180.0]


def test_a_point_in_range_is_left_as_it_is() -> None:
    """Both ends of the range count as in range: 180 does not become -180."""
    assert _x(_points(180.0, -180.0, 0.0, 179.5)) == [180.0, -180.0, 0.0, 179.5]


def test_an_empty_point_stays_empty() -> None:
    (x,) = _x(_points(math.nan))

    assert math.isnan(x)


@pytest.mark.parametrize("skip_crossing", [False, True])
def test_a_geometry_entirely_past_the_antimeridian_moves_whole(
    skip_crossing: bool,
) -> None:
    lines = _lines([190.0, 200.0], [-190.0, -200.0])

    assert _x(lines, skip_crossing=skip_crossing) == [
        [-170.0, -160.0],
        [170.0, 160.0],
    ]


def test_a_crossing_geometry_is_torn_without_the_flag() -> None:
    assert _x(_lines([170.0, 190.0])) == [[170.0, -170.0]]


def test_a_crossing_geometry_is_left_as_it_is_with_the_flag() -> None:
    assert _x(_lines([170.0, 190.0]), skip_crossing=True) == [[170.0, 190.0]]


def test_a_geometry_spanning_more_than_a_turn_is_crossing() -> None:
    """Every vertex is past the antimeridian, but not by the same turns."""
    assert _x(_lines([190.0, 600.0]), skip_crossing=True) == [[190.0, 600.0]]


def test_the_flag_decides_per_geometry() -> None:
    lines = _lines([170.0, 190.0], [190.0, 200.0], [0.0, 10.0])

    assert _x(lines, skip_crossing=True) == [
        [170.0, 190.0],
        [-170.0, -160.0],
        [0.0, 10.0],
    ]


def test_a_box_polygon_across_the_antimeridian_is_crossing() -> None:
    """`to_polygon` pushes it to 190, which is exactly what the flag protects."""
    polygons = (
        pl.LazyFrame({"a": [170.0], "b": [-10.0], "c": [-170.0], "d": [10.0]})
        .select(geo.box("a", "b", "c", "d", crs=WGS84).alias("box"))
        .select(geo.to_polygon("box"))
    )
    assert _x(polygons, skip_crossing=True) == [[[170.0, 190.0, 190.0, 170.0, 170.0]]]
    assert _x(polygons) == [[[170.0, -170.0, -170.0, 170.0, 170.0]]]


def test_every_ring_of_a_polygon_counts() -> None:
    """The exterior is past the antimeridian, the hole is not: crossing."""
    polygons = pl.LazyFrame(
        {
            "exterior_x": [[190.0, 200.0, 200.0, 190.0]],
            "hole_x": [[170.0, 175.0, 175.0, 170.0]],
            "y": [[0.0, 0.0, 1.0, 0.0]],
        }
    ).select(
        geo.polygons_from_rings_in_separate_columns(
            geo.line_string_from_columns("exterior_x", "y", crs=WGS84),
            geo.line_string_from_columns("hole_x", "y", crs=WGS84),
        ).alias("polygon")
    )

    assert _x(polygons, skip_crossing=True) == [
        [[190.0, 200.0, 200.0, 190.0], [170.0, 175.0, 175.0, 170.0]]
    ]
    assert _x(polygons) == [
        [[-170.0, -160.0, -160.0, -170.0], [170.0, 175.0, 175.0, 170.0]]
    ]


def test_every_part_of_a_multipolygon_counts() -> None:
    ring = [0.0, 0.0, 1.0, 0.0]
    multipolygons = pl.LazyFrame(
        {
            "x": [
                # One part on each side: crossing.
                [[[170.0, 175.0, 175.0, 170.0]], [[190.0, 195.0, 195.0, 190.0]]],
                # Both parts past it: moves whole.
                [[[190.0, 195.0, 195.0, 190.0]], [[200.0, 205.0, 205.0, 200.0]]],
            ],
            "y": [[[ring], [ring]], [[ring], [ring]]],
        }
    ).select(
        geo.multi_polygons_from_nested_coordinates("x", "y", crs=WGS84).alias("parts")
    )

    assert _x(multipolygons, skip_crossing=True) == [
        [[[170.0, 175.0, 175.0, 170.0]], [[190.0, 195.0, 195.0, 190.0]]],
        [[[-170.0, -165.0, -165.0, -170.0]], [[-160.0, -155.0, -155.0, -160.0]]],
    ]


def _xyzm_point() -> pl.LazyFrame:
    return pl.LazyFrame(
        {"x": [190.0], "y": [200.0], "z": [500.0], "m": [-400.0]}
    ).select(geo.point("x", "y", "z", "m", crs=WGS84).alias("point"))


def test_only_x_moves() -> None:
    out = _xyzm_point().select(geo.wrap_longitude("point").ext.storage()).collect()

    assert out.item() == {"x": -170.0, "y": 200.0, "z": 500.0, "m": -400.0}


@pytest.mark.parametrize("skip_crossing", [False, True])
def test_the_dtype_and_crs_are_kept(skip_crossing: bool) -> None:
    lines = _lines([170.0, 190.0]).select(
        geo.wrap_longitude("line", skip_crossing=skip_crossing)
    )
    points = _xyzm_point().select(
        geo.wrap_longitude("point", skip_crossing=skip_crossing)
    )

    assert lines.collect_schema()["line"] == LineStringXY(crs=WGS84)
    assert lines.collect().schema["line"] == LineStringXY(crs=WGS84)
    assert points.collect_schema()["point"] == PointXYZM(crs=WGS84)


@pytest.mark.parametrize("skip_crossing", [False, True])
def test_null_and_empty_geometries_pass_through(skip_crossing: bool) -> None:
    lines = pl.LazyFrame(
        {"x": [None, [], [190.0]], "y": [None, [], [0.0]]},
        schema={"x": pl.List(pl.Float64), "y": pl.List(pl.Float64)},
    ).select(geo.line_string_from_columns("x", "y", crs=WGS84).alias("line"))

    assert _x(lines, skip_crossing=skip_crossing) == [None, [], [-170.0]]


def test_start_moves_the_range() -> None:
    """0..360 data: -10 is 350, 370 is 10, and both ends stay where they are."""
    points = _points(-10.0, 370.0, 0.0, 360.0, 190.0, -370.0)

    assert _x(points, start=0.0) == [350.0, 10.0, 0.0, 360.0, 190.0, 350.0]


def test_start_moves_what_counts_as_crossing() -> None:
    """170..190 lies inside 0..360, and -10..10 crosses its edge at 0."""
    lines = _lines([170.0, 190.0], [-10.0, 10.0], [-20.0, -10.0])

    assert _x(lines, start=0.0, skip_crossing=True) == [
        [170.0, 190.0],
        [-10.0, 10.0],
        [340.0, 350.0],
    ]


def test_start_is_in_the_unit_of_the_crs() -> None:
    """Grads: 0..400."""
    assert _x(_points(-10.0, 410.0, crs="EPSG:4807"), start=0.0) == [390.0, 10.0]


@pytest.mark.parametrize("start", [math.nan, math.inf])
def test_start_has_to_be_finite(start: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        geo.wrap_longitude("point", start=start)


def test_grads_have_a_turn_of_400() -> None:
    """NTF (Paris) is in grads: the range is [-200, 200]."""
    assert _x(_points(190.0, 210.0, -250.0, crs="EPSG:4807")) == [190.0, -190.0, 150.0]


@pytest.mark.parametrize("crs", [None, "EPSG:28992"], ids=["no crs", "projected"])
def test_without_a_longitude_it_is_refused_at_plan_time(crs: str | None) -> None:
    with pytest.raises(TypeError, match="needs a geographic CRS"):
        _points(190.0, crs=crs).select(geo.wrap_longitude("point")).collect_schema()


def test_a_box_is_refused_at_plan_time() -> None:
    boxes = pl.LazyFrame({"a": [0.0], "b": [0.0], "c": [1.0], "d": [1.0]}).select(
        geo.box("a", "b", "c", "d", crs=WGS84).alias("box")
    )

    with pytest.raises(TypeError, match="got: BoxXY"):
        boxes.select(geo.wrap_longitude("box")).collect_schema()


def test_namespace_matches_the_functional_api() -> None:
    lines = _lines([170.0, 190.0], [190.0, 200.0])

    for skip_crossing in (False, True):
        functional = lines.select(
            geo.wrap_longitude("line", start=0.0, skip_crossing=skip_crossing)
        )
        namespace = lines.select(
            gpl.col("line").geo.wrap_longitude(start=0.0, skip_crossing=skip_crossing)
        )
        assert namespace.collect().equals(functional.collect())
