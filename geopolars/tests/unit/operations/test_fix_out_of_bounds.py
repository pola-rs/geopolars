"""Dropping or clamping the coordinates outside the CRS's bounds."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYZM,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PolygonXY,
)

WGS84 = "EPSG:4326"
UTM31N = "EPSG:32631"
# NTF (Paris): latitude first, in grads.
NTF_PARIS = "EPSG:4807"

NAN, INF = float("nan"), float("inf")

Coord = dict[str, float]
Line = list[Coord]


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


def _frame(dtype: GeoArrowType, *rows: object) -> pl.DataFrame:
    """`rows`, relabelled as `dtype` without any checks."""
    storage = pl.Series("g", list(rows), dtype=dtype._geo_storage)
    return pl.DataFrame(storage.ext.to(dtype))


def _nan_as_str(value: object) -> object:
    """`value`, with every NaN as `"nan"`, so that lists of them compare equal."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [_nan_as_str(item) for item in value]
    if isinstance(value, dict):
        return {key: _nan_as_str(item) for key, item in value.items()}
    return value


def _fix(dtype: GeoArrowType, *rows: object, **kwargs: object) -> list:
    out = _frame(dtype, *rows).select(geo.fix_out_of_bounds("g", **kwargs))
    return _nan_as_str(out.select(pl.col("g").ext.storage())["g"].to_list())  # type: ignore[return-value]


_A, _B = _line((1, 1)), _line((2, 2))
_NORTH, _EAST = _line((0, 95)), _line((200, 0))
_EMPTY = {"x": "nan", "y": "nan"}


def test_the_vertices_outside_are_dropped() -> None:
    out = _fix(LineStringXY(crs=WGS84), _A + _NORTH + _B + _EAST)

    assert out == [_A + _B]


def test_the_vertices_outside_are_clamped() -> None:
    out = _fix(LineStringXY(crs=WGS84), _A + _NORTH + _B + _EAST, action="clamp")

    assert out == [_A + _line((0, 90)) + _B + _line((180, 0))]


def test_the_vertices_outside_are_emptied() -> None:
    out = _fix(LineStringXY(crs=WGS84), _A + _NORTH + _B, action="set_empty")

    assert out == [[*_A, _EMPTY, *_B]]


def test_clamping_keeps_z_and_m() -> None:
    line = [{"x": 200.0, "y": -95.0, "z": 1.0, "m": 2.0}]

    out = _fix(LineStringXYZM(crs=WGS84), line, action="clamp")

    assert out == [[{"x": 180.0, "y": -90.0, "z": 1.0, "m": 2.0}]]


@pytest.mark.parametrize(
    ("action", "fixed"),
    [
        ("drop", _EMPTY),
        ("set_empty", _EMPTY),
        ("set_null", None),
        ("clamp", {"x": 0.0, "y": 90.0}),
    ],
)
def test_a_point_outside_is_fixed(action: str, fixed: Coord | None) -> None:
    rows = [*_A, *_NORTH, *_line((NAN, NAN)), None]

    assert _fix(PointXY(crs=WGS84), *rows, action=action) == [
        *_A,
        fixed,
        _EMPTY,
        None,
    ]


def test_a_nan_beside_a_number_is_dropped_but_not_clamped() -> None:
    line = _A + _line((NAN, 0)) + _B

    assert _fix(LineStringXY(crs=WGS84), line) == [_A + _B]
    assert _fix(LineStringXY(crs=WGS84), line, action="clamp") == [_nan_as_str(line)]


def test_an_infinity_is_clamped_onto_the_bound() -> None:
    out = _fix(PointXY(crs=WGS84), *_line((INF, -INF)), action="clamp")

    assert out == _line((180, -90))


def test_a_multipoint_can_lose_all_of_its_points() -> None:
    assert _fix(MultiPointXY(crs=WGS84), _A + _NORTH, _EAST) == [_A, []]


@pytest.mark.parametrize(
    ("short_lines", "expected"),
    [("null", None), ("keep", [_A + _B, _A]), ("empty", [_A + _B, []])],
)
def test_a_short_part_of_a_multilinestring(short_lines: str, expected: object) -> None:
    out = _fix(
        MultiLineStringXY(crs=WGS84), [_A + _B, _A + _NORTH], short_lines=short_lines
    )

    assert out == [expected]


_SQUARE = _line((0, 0), (4, 0), (4, 4), (0, 4), (0, 0))
_SPIKE = _line((0, 0), (4, 0), (4, 95), (0, 4), (0, 0))
_OVER_THE_POLE = _line((0, 95), (4, 0), (4, 4), (0, 4), (0, 95))


def test_a_ring_keeps_its_vertices_inside() -> None:
    out = _fix(PolygonXY(crs=WGS84), [_SPIKE])

    assert out == [[_line((0, 0), (4, 0), (0, 4), (0, 0))]]


@pytest.mark.parametrize(
    ("short_lines", "expected"),
    [
        ("null", None),
        ("keep", [_line((4, 0), (4, 4), (0, 4))]),
        ("empty", [[]]),
    ],
)
def test_a_ring_that_loses_its_ends_is_too_short(
    short_lines: str, expected: object
) -> None:
    assert _fix(PolygonXY(crs=WGS84), [_OVER_THE_POLE], short_lines=short_lines) == [
        expected
    ]


def test_a_clamped_ring_stays_closed() -> None:
    out = _fix(PolygonXY(crs=WGS84), [_OVER_THE_POLE], action="clamp")

    assert out == [[_line((0, 90), (4, 0), (4, 4), (0, 4), (0, 90))]]


def test_only_the_multipolygon_with_a_short_ring_becomes_null() -> None:
    out = _fix(MultiPolygonXY(crs=WGS84), [[_SQUARE], [_OVER_THE_POLE]], [[_SQUARE]])

    assert out == [None, [[_SQUARE]]]


def test_only_the_rows_with_a_vertex_outside_become_null() -> None:
    out = _fix(
        MultiPolygonXY(crs=WGS84),
        [[_SQUARE], [_SPIKE]],
        [[_SQUARE]],
        None,
        action="set_null",
    )

    assert out == [None, [[_SQUARE]], None]


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("drop", _line((270, 0), (360, 0))),
        ("clamp", _line((270, 0), (360, 0), (360, 0), (-180, 0))),
    ],
)
def test_a_wrapped_longitude_can_be_allowed(action: str, expected: Line) -> None:
    line = _line((270, 0), (360, 0), (361, 0), (-200, 0))

    out = _fix(
        LineStringXY(crs=WGS84),
        line,
        crs_bounds="crs_wrapped_longitude",
        action=action,
        short_lines="keep",
    )

    assert out == [expected]


def test_the_area_of_use_narrows_a_geographic_crs() -> None:
    """ETRS89 is meant for Europe only."""
    points = _line((5, 52), (-100, 40))

    out = _fix(PointXY(crs="EPSG:4258"), *points, crs_bounds="area_of_use")

    assert out == [points[0], _EMPTY]


def test_the_bounds_are_in_the_unit_of_each_axis() -> None:
    """NTF (Paris) declares latitude first, in grads: `y` runs to ±100, `x` to ±200."""
    points = _line((0, 95), (0, 101), (210, 0))
    dtype = PointXY(crs=NTF_PARIS)

    assert _fix(dtype, *points, action="clamp") == _line((0, 95), (0, 100), (200, 0))
    valid = _frame(dtype, *points).select(
        geo.is_valid_geoarrow("g", out_of_bounds=geo.OutOfBounds())
    )
    assert valid["g"].to_list() == [True, False, False]


@pytest.mark.parametrize("action", ["drop", "clamp"])
def test_the_result_is_within_the_bounds(action: str) -> None:
    out_of_bounds = geo.OutOfBounds(action=action)  # type: ignore[arg-type]
    df = _frame(LineStringXY(crs=WGS84), _A + _NORTH + _B + _EAST, _A + _B)

    valid = df.select(
        geo.fix_out_of_bounds("g", action=action).geo.is_valid_geoarrow(  # type: ignore[arg-type]
            out_of_bounds=out_of_bounds
        )
    )

    assert valid["g"].to_list() == [True, True]


@pytest.mark.parametrize("dtype", [LineStringXY(), LineStringXY(crs=UTM31N)])
def test_without_bounds_the_input_is_returned(dtype: GeoArrowType) -> None:
    line = _line((1e7, 1e8), (-1e7, 0))

    assert _fix(dtype, line, action="clamp") == [line]


def test_a_null_row_stays_null() -> None:
    assert _fix(LineStringXY(crs=WGS84), None, _A + _NORTH + _B) == [None, _A + _B]


def test_the_crs_is_carried_over() -> None:
    df = _frame(PolygonXY(crs=WGS84), [_SPIKE])

    assert df.select(geo.fix_out_of_bounds("g")).schema == df.schema


def test_namespace_matches_the_functional_api() -> None:
    df = _frame(LineStringXY(crs=WGS84), _A + _NORTH + _B)

    assert_frame_equal(
        df.select(gpl.col("g").geo.fix_out_of_bounds(action="clamp")),
        df.select(geo.fix_out_of_bounds("g", action="clamp")),
    )


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"crs_bounds": "off"}, "crs_bounds"),
        ({"action": "keep"}, "action"),
        ({"short_lines": "drop"}, "short_lines"),
    ],
)
def test_an_unknown_option_is_refused(kwargs: dict[str, str], name: str) -> None:
    with pytest.raises(ValueError, match=name):
        geo.fix_out_of_bounds("g", **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=name):
        geo.OutOfBounds(**kwargs)  # type: ignore[arg-type]
