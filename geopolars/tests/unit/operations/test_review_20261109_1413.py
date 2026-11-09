"""Reproducers for the review of the hausdorff branch. Not committed."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.exceptions import ComputeError

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    PointXY,
    PointXYZ,
    PolygonXY,
)
from tests.unit.operations.test_hausdorff_line_between import (
    _as_wkt,
    _hausdorff_lines,
    _line_strings,
    _multi_points,
    _wkt,
)


def _with_crs(
    name: str, dtype: GeoArrowType, wkts: list[str | None], crs: str
) -> pl.Series:
    return pl.select(geo.set_crs(_wkt(name, dtype, wkts), crs)).to_series()


@pytest.mark.parametrize("lat", [90, 91])
def test_arch1_ignore_per_row_errors_survives_a_bad_latitude_under_default_in_crs(
    lat: int,
) -> None:
    df = pl.DataFrame(
        [
            _with_crs("a", PointXY(), [f"POINT (0 {lat})", "POINT (0 0)"], "EPSG:4326"),
            _with_crs("b", PointXY(), ["POINT (1 0)", "POINT (1 0)"], "EPSG:4326"),
        ]
    )
    out = df.select(geo.shortest_line_between("a", "b", ignore_per_row_errors=True))
    assert out["a"].null_count() == 1


def test_arch5_nan_z_not_in_output_does_not_raise() -> None:
    a = _line_strings("a", [[(0, 0, math.nan), (1, 0, 0)]], axes="xyz")
    b = _wkt("b", PointXY(), ["POINT (1 1)"])
    out = pl.DataFrame([a, b]).select(geo.hausdorff_line_between("a", "b"))
    assert out.schema["a"] == LineStringXY()


@pytest.mark.parametrize("empty_first", [True, False])
def test_tests1_skip_empty_only_empty_points_is_empty_against_a_failing_one(
    empty_first: bool,
) -> None:
    a = _multi_points("a", [[(math.nan, math.nan)]])
    b = _line_strings("b", [[(0, 0), (math.nan, 1)]])
    x, y = ("a", "b") if empty_first else ("b", "a")
    out = pl.DataFrame([a, b]).select(
        geo.hausdorff_line_between(x, y, non_finite="drop_empty")
    )
    assert _as_wkt(out[x]) == ["LINESTRING EMPTY"]


@pytest.mark.parametrize(
    ("dtype", "wkt"),
    [
        (LineStringXY(), "LINESTRING (0 0, 1 0)"),
        (PointXY(), "POINT (2 3)"),
        (PolygonXY(), "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
    ],
)
def test_tests3_equal_geometries_give_a_line_of_length_zero(
    dtype: GeoArrowType, wkt: str
) -> None:
    [line] = _hausdorff_lines((dtype, [wkt]), (dtype, [wkt]))
    assert line is not None
    assert line != "LINESTRING EMPTY"


def test_tests9c_row_beyond_the_pole_with_ignore_per_row_errors() -> None:
    df = pl.DataFrame(
        [
            _with_crs("a", PointXY(), ["POINT (0 91)"], "EPSG:4326"),
            _with_crs("b", PointXY(), ["POINT (0 0)"], "EPSG:4326"),
        ]
    )
    out = df.select(geo.shortest_line_between("a", "b", ignore_per_row_errors=True))
    assert out["a"].to_list() == [None]


def test_tests9a_default_in_crs_keeps_nulls_and_empties() -> None:
    many = _with_crs(
        "many", PointXY(), ["POINT (3 60)", None, "POINT EMPTY"], "EPSG:4326"
    )
    one = _with_crs(
        "one", PolygonXY(), ["POLYGON ((0 59, 1 59, 1 61, 0 61, 0 59))"], "EPSG:4326"
    )
    df = pl.DataFrame(many)
    out = df.select(geo.shortest_line_between("many", one))
    assert _as_wkt(out["many"])[1:] == [None, "LINESTRING EMPTY"]
    out = df.select(geo.shortest_line_between(one, "many"))
    assert _as_wkt(out.to_series())[1:] == [None, "LINESTRING EMPTY"]


def test_tests9b_z_reprojected_to_a_crs_with_z() -> None:
    df = pl.DataFrame(
        [
            _with_crs("a", PointXYZ(), ["POINT Z (0 60 100)"], "EPSG:4326"),
            _with_crs("b", PointXY(), ["POINT (1 60)"], "EPSG:4326"),
        ]
    )
    out = df.select(geo.shortest_line_between("a", "b", in_crs="EPSG:4979"))
    assert out.schema["a"] == LineStringXY(crs="EPSG:4979")


def test_tests5c_ignore_z_does_not_make_a_vertex_the_empty_point() -> None:
    b = _line_strings("b", [[(0, 1), (5, 4), (10, 1)]])
    a = _line_strings(
        "a", [[(0, 0, 1), (math.nan, math.nan, 5), (10, 0, 1)]], axes="xyz"
    )
    with pytest.raises(ComputeError, match="expects finite coordinates"):
        pl.DataFrame([a, b]).select(
            geo.hausdorff_line_between("a", "b", non_finite="drop_empty", ignore_z=True)
        )
