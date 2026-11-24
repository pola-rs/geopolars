"""Turning a `geoarrow.box` into the polygon it describes."""

from __future__ import annotations

import math

import polars as pl
import pytest

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    BoxType,
    BoxXY,
    BoxXYM,
    BoxXYZ,
    BoxXYZM,
    PolygonType,
    PolygonXY,
    PolygonXYM,
    PolygonXYZ,
    PolygonXYZM,
)


def _polygons(*bounds: tuple[float | None, ...], crs: str | None = None) -> pl.Series:
    """One `to_polygon` per row of `(xmin, ymin, xmax, ymax)` bounds."""
    df = pl.DataFrame(
        list(bounds), schema=["xmin", "ymin", "xmax", "ymax"], orient="row"
    )
    return (
        df.lazy()
        .select(geo.box("xmin", "ymin", "xmax", "ymax", crs=crs).alias("box"))
        .select(geo.to_polygon("box"))
        .collect()
        .to_series()
    )


def _rings(polygons: pl.Series) -> list:
    return polygons.ext.storage().to_list()


def test_a_box_is_one_closed_counter_clockwise_ring() -> None:
    polygons = _polygons((0.0, 1.0, 2.0, 3.0))

    assert polygons.dtype == PolygonXY()
    assert _rings(polygons) == [
        [
            [
                {"x": 0.0, "y": 1.0},
                {"x": 2.0, "y": 1.0},
                {"x": 2.0, "y": 3.0},
                {"x": 0.0, "y": 3.0},
                {"x": 0.0, "y": 1.0},
            ]
        ]
    ]


def test_the_result_is_named_after_the_input() -> None:
    assert _polygons((0.0, 0.0, 1.0, 1.0)).name == "box"


def test_the_crs_is_kept() -> None:
    polygons = _polygons((0.0, 0.0, 1.0, 1.0), crs="EPSG:28992")

    assert polygons.dtype == PolygonXY(crs="EPSG:28992")


def test_the_polygon_is_a_geometry_like_any_other() -> None:
    """The point of the conversion: other operations accept the result."""
    df = _polygons((0.0, 0.0, 4.0, 2.0)).to_frame()

    assert df.select(geo.area("box")).item() == 8.0


def test_a_null_box_gives_a_null_polygon() -> None:
    assert _rings(_polygons((None, 0.0, 1.0, 1.0))) == [None]


@pytest.mark.parametrize(
    "bounds",
    [
        (math.inf, 0.0, -math.inf, 1.0),
        (0.0, math.inf, 1.0, -math.inf),
        (math.inf, math.inf, -math.inf, -math.inf),
        # Unlike x, y has no antimeridian: a reversed range holds nothing.
        (0.0, 1.0, 1.0, 0.0),
    ],
    ids=["x", "y", "both", "reversed y"],
)
def test_an_empty_range_gives_an_empty_polygon(bounds: tuple[float, ...]) -> None:
    polygons = _polygons(bounds)

    assert _rings(polygons) == [[]]
    assert polygons.to_frame().select(geo.is_empty("box")).item()


@pytest.mark.parametrize("crs", [None, "EPSG:28992"], ids=["no crs", "projected"])
def test_a_box_across_the_antimeridian_without_a_longitude_is_null(
    crs: str | None,
) -> None:
    """Without a longitude there is no way around: no polygon describes the box."""
    assert _rings(_polygons((170.0, -10.0, -170.0, 10.0), crs=crs)) == [None]


@pytest.mark.parametrize(
    ("crs", "turn"),
    [
        ("EPSG:4326", 360.0),
        # WGS 84 with EGM96 heights: x and y are still longitude and latitude.
        ("EPSG:4326+5773", 360.0),
        # NTF (Paris), in grads.
        ("EPSG:4807", 400.0),
    ],
    ids=["degrees", "compound", "grads"],
)
def test_a_geographic_box_continues_past_the_antimeridian(
    crs: str, turn: float
) -> None:
    """The polygon covers the side the box means:
    from 170 east to -170 is from 170 to 190, not back west to -170."""
    (polygon,) = _rings(_polygons((170.0, -10.0, -170.0, 10.0), crs=crs))

    assert [(vertex["x"], vertex["y"]) for vertex in polygon[0]] == [
        (170.0, -10.0),
        (turn - 170.0, -10.0),
        (turn - 170.0, 10.0),
        (170.0, 10.0),
        (170.0, -10.0),
    ]


def test_a_geographic_box_measures_the_side_it_means() -> None:
    """20 degrees across the antimeridian is as large as 20 degrees anywhere."""
    across = _polygons((170.0, -10.0, -170.0, 10.0), crs="EPSG:4326").to_frame()
    meridian = _polygons((-10.0, -10.0, 10.0, 10.0), crs="EPSG:4326").to_frame()

    assert across.select(geo.area("box")).item() == pytest.approx(
        meridian.select(geo.area("box")).item()
    )


def test_a_geographic_box_that_does_not_cross_is_left_as_it_is() -> None:
    (polygon,) = _rings(_polygons((-170.0, -10.0, 170.0, 10.0), crs="EPSG:4326"))

    assert {vertex["x"] for vertex in polygon[0]} == {-170.0, 170.0}


def test_an_empty_geographic_box_stays_empty() -> None:
    """`inf` to `-inf` also has `xmin > xmax`, but is not a box across anything."""
    bounds = (math.inf, -10.0, -math.inf, 10.0)

    assert _rings(_polygons(bounds, crs="EPSG:4326")) == [[]]


def test_a_crs_proj_cannot_read_is_refused() -> None:
    """Whether the box can go around depends on the CRS, so it has to be readable."""
    with pytest.raises(pl.exceptions.ComputeError, match="PROJ cannot use the CRS"):
        _polygons((0.0, 0.0, 1.0, 1.0), crs="not-a-crs")


def test_mixed_rows_are_handled_per_row() -> None:
    rings = _rings(
        _polygons(
            (0.0, 0.0, 1.0, 1.0),
            (170.0, 0.0, -170.0, 1.0),
            (math.inf, 0.0, -math.inf, 1.0),
            (None, None, None, None),
        )
    )

    assert [len(polygon) if polygon is not None else None for polygon in rings] == [
        1,
        None,
        0,
        None,
    ]


def _boxes_of(dtype: type[BoxType], **bounds: list[float]) -> pl.LazyFrame:
    names = list(dtype().ext_storage().to_schema())
    return pl.LazyFrame({name: bounds[name] for name in names}).select(
        pl.struct(names).ext.to(dtype(crs="EPSG:4326")).alias("box")
    )


_XYZM = {
    "xmin": [0.0],
    "ymin": [1.0],
    "zmin": [2.0],
    "mmin": [3.0],
    "xmax": [10.0],
    "ymax": [11.0],
    "zmax": [12.0],
    "mmax": [13.0],
}


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (BoxXY, PolygonXY),
        (BoxXYZ, PolygonXYZ),
        (BoxXYM, PolygonXYM),
        (BoxXYZM, PolygonXYZM),
    ],
    ids=["xy", "xyz", "xym", "xyzm"],
)
def test_the_polygon_has_the_boxs_dimension_and_crs(
    dtype: type[BoxType], expected: type[PolygonType]
) -> None:
    lf = _boxes_of(dtype, **_XYZM).select(geo.to_polygon("box"))

    assert lf.collect_schema()["box"] == expected(crs="EPSG:4326")
    assert lf.collect().schema["box"] == expected(crs="EPSG:4326")


def test_z_and_m_follow_x() -> None:
    """The vertices at xmin take the minima, those at xmax the maxima,
    so both ends of every range survive."""
    (polygon,) = (
        _boxes_of(BoxXYZM, **_XYZM)
        .select(geo.to_polygon("box").ext.storage())
        .collect()
        .to_series()
        .to_list()
    )

    assert [tuple(vertex.values()) for vertex in polygon[0]] == [
        (0.0, 1.0, 2.0, 3.0),
        (10.0, 1.0, 12.0, 13.0),
        (10.0, 11.0, 12.0, 13.0),
        (0.0, 11.0, 2.0, 3.0),
        (0.0, 1.0, 2.0, 3.0),
    ]


@pytest.mark.parametrize("axis", ["z", "m"])
@pytest.mark.parametrize(
    ("low", "high"), [(math.inf, -math.inf), (1.0, 0.0)], ids=["inf", "reversed"]
)
def test_an_empty_z_or_m_range_gives_an_empty_polygon(
    axis: str, low: float, high: float
) -> None:
    """Only x has the antimeridian exception: any other reversed range holds nothing."""
    bounds = {**_XYZM, f"{axis}min": [low], f"{axis}max": [high]}
    out = _boxes_of(BoxXYZM, **bounds).select(geo.to_polygon("box")).collect()

    assert out.select(pl.col("box").ext.storage()).item().to_list() == []


def test_a_z_box_continues_past_the_antimeridian() -> None:
    """The turn only moves x: z still follows it."""
    bounds = {**_XYZM, "xmin": [170.0], "xmax": [-170.0]}
    (polygon,) = (
        _boxes_of(BoxXYZ, **bounds)
        .select(geo.to_polygon("box").ext.storage())
        .collect()
        .to_series()
        .to_list()
    )

    assert [(vertex["x"], vertex["z"]) for vertex in polygon[0]] == [
        (170.0, 2.0),
        (190.0, 12.0),
        (190.0, 12.0),
        (170.0, 2.0),
        (170.0, 2.0),
    ]


def test_a_geometry_is_refused_at_plan_time() -> None:
    lf = pl.LazyFrame({"x": [0.0], "y": [0.0]}).select(
        geo.point("x", "y").alias("point")
    )

    with pytest.raises(TypeError, match="expected a `geoarrow.box` column"):
        lf.select(geo.to_polygon("point")).collect_schema()


def test_namespace_matches_the_functional_api() -> None:
    df = _polygons((0.0, 0.0, 1.0, 1.0)).to_frame()
    boxes = pl.DataFrame({"a": [0.0], "b": [0.0], "c": [1.0], "d": [1.0]}).select(
        geo.box("a", "b", "c", "d").alias("box")
    )

    assert boxes.select(gpl.col("box").geo.to_polygon()).equals(df)
