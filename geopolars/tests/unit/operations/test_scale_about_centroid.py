"""Unit tests for `geo.scale_about_centroid`."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from tests.unit.conftest import XY, XYZ, XYZM, Dimension, coordinates

RD = "EPSG:28992"


def test_a_polygon_scales_about_its_centroid(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords)

    assert_frame_equal(
        df.select(geo.scale_about_centroid("polygon", 2.0, 0.5)),
        df.select(geo.scale("polygon", 2.0, 0.5, origin=geo.centroid("polygon"))),
        check_exact=True,
    )


def test_a_line_scales_about_its_centroid(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.lines(line_coords)

    assert_frame_equal(
        df.select(geo.scale_about_centroid("line", 2.0, 0.5)),
        df.select(geo.scale("line", 2.0, 0.5, origin=geo.centroid("line"))),
        check_exact=True,
    )


@pytest.mark.parametrize("dimension", [XYZ, XYZM], ids=["XYZ", "XYZM"])
def test_a_line_scales_z_about_its_centroid(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A line's centroid keeps its `z`, so `z` can be scaled about it too."""
    df = dimension.lines(line_coords)

    assert_frame_equal(
        df.select(geo.scale_about_centroid("line", zfact=3.0)),
        df.select(geo.scale("line", zfact=3.0, origin=geo.centroid("line"))),
        check_exact=True,
    )


def test_the_centroid_stays_where_it_is(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords)
    scaled = df.select(geo.scale_about_centroid("polygon", 2.0, 3.0))

    assert_frame_equal(
        coordinates(scaled.select(geo.centroid("polygon")), "polygon"),
        coordinates(df.select(geo.centroid("polygon")), "polygon"),
    )


def test_each_geometry_scales_about_its_own_centroid(
    ring_coords: pl.DataFrame,
) -> None:
    df = XY.polygons(ring_coords)
    out = df.select(geo.scale_about_centroid("polygon", 2.0, 0.5))

    for row in range(df.height):
        alone = df.slice(row, 1).select(geo.scale_about_centroid("polygon", 2.0, 0.5))
        assert_frame_equal(out.slice(row, 1), alone, check_exact=True)


@pytest.mark.parametrize("dimension", [XYZ, XYZM], ids=["XYZ", "XYZM"])
def test_a_polygon_scales_z_about_its_centroid(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon's centroid keeps its `z`, so `z` can be scaled about it too."""
    df = dimension.polygons(ring_coords)

    assert_frame_equal(
        df.select(geo.scale_about_centroid("polygon", zfact=3.0)),
        df.select(geo.scale("polygon", zfact=3.0, origin=geo.centroid("polygon"))),
        check_exact=True,
    )


def test_zfact_without_a_z_is_refused(ring_coords: pl.DataFrame) -> None:
    """The same refusal as `scale`, rather than one about the centroid."""
    out = XY.polygons(ring_coords).lazy()

    with pytest.raises(TypeError, match="has no z coordinate"):
        out.select(geo.scale_about_centroid("polygon", zfact=2.0)).collect_schema()


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
        with pytest.raises(TypeError, match="cannot scale about its centroid"):
            df.lazy().select(geo.scale_about_centroid(name, 2.0)).collect_schema()


def test_bad_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="`yfact`"):
        geo.scale_about_centroid("line", 2.0, float("nan"))


def test_a_null_comes_out_null_and_an_empty_geometry_stays_empty() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 0.0}], [], None]},
        schema={"line": pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))},
    ).select(pl.col("line").ext.to(XY.linestring_dtype()))
    out = df.select(geo.scale_about_centroid("line", 2.0))

    assert out["line"].ext.storage()[1:].to_list() == [[], None]
    assert out["line"].ext.storage()[0].to_list() == [
        {"x": -1.0, "y": 0.0},
        {"x": 3.0, "y": 0.0},
    ]


def test_a_line_without_length_goes_to_on_row_error() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 1.0, "y": 1.0}, {"x": 1.0, "y": 1.0}]]},
        schema={"line": pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))},
    ).select(pl.col("line").ext.to(XY.linestring_dtype()))

    with pytest.raises(pl.exceptions.ComputeError, match="got one without"):
        df.select(geo.scale_about_centroid("line", 2.0))
    out = df.select(geo.scale_about_centroid("line", 2.0, on_row_error="null"))

    assert out["line"].is_null().to_list() == [True]


def test_the_crs_is_carried_over(ring_coords: pl.DataFrame) -> None:
    df = XY.polygons(ring_coords).select(
        pl.col("polygon").ext.storage().ext.to(XY.polygon_dtype(crs=RD))
    )
    out = df.lazy().select(geo.scale_about_centroid("polygon", 2.0))

    assert out.collect_schema()["polygon"] == df.schema["polygon"]
    out.collect()


def test_the_namespace_forwards(line_coords: pl.DataFrame) -> None:
    df = XYZ.lines(line_coords)

    assert_frame_equal(
        df.select(gpl.col("line").geo.scale_about_centroid(2.0, 0.5, 3.0)),
        df.select(geo.scale_about_centroid("line", 2.0, 0.5, 3.0)),
    )
