"""Emptying or dropping the coordinates that are not finite, or empty."""

from __future__ import annotations

import math

import polars as pl
import pytest

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
    PolygonXYZM,
)

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


def _filter(dtype: GeoArrowType, *rows: object, **kwargs: object) -> list:
    out = _frame(dtype, *rows).select(geo.filter_coordinates("g", **kwargs))
    return _nan_as_str(out.select(pl.col("g").ext.storage())["g"].to_list())  # type: ignore[return-value]


_SQUARE = _line((0, 0), (4, 0), (4, 4), (0, 4), (0, 0))
_EMPTY = {"x": "nan", "y": "nan"}
_A, _B = _line((1, 1)), _line((2, 2))
_NOT_FINITE = _line((NAN, 1)) + _line((1, INF))
_NOT_FINITE_OUT = [{"x": "nan", "y": 1.0}, {"x": 1.0, "y": INF}]
_EMPTY_POINT = _line((NAN, NAN))


@pytest.mark.parametrize(
    ("select_non_finite", "select_empty", "expected"),
    [
        (True, True, _A + _B),
        (True, False, [*_A, _EMPTY, *_B]),
        (False, True, _A + _NOT_FINITE_OUT + _B),
        (False, False, [*_A, *_NOT_FINITE_OUT, _EMPTY, *_B]),
    ],
)
def test_the_selected_vertices_are_dropped(
    select_non_finite: bool, select_empty: bool, expected: Line
) -> None:
    line = _A + _NOT_FINITE + _EMPTY_POINT + _B

    out = _filter(
        LineStringXY(),
        line,
        select_non_finite=select_non_finite,
        select_empty=select_empty,
    )

    assert out == [expected]


def test_the_selected_vertices_are_emptied() -> None:
    line = _A + _NOT_FINITE + _EMPTY_POINT + _B

    out = _filter(LineStringXY(), line, action="set_empty")

    assert out == [_A + [_EMPTY, _EMPTY, _EMPTY] + _B]


def test_emptying_only_empty_vertices_returns_the_input() -> None:
    line = _A + _NOT_FINITE + _EMPTY_POINT + _B

    out = _filter(LineStringXY(), line, action="set_empty", select_non_finite=False)

    assert out == [_nan_as_str(line)]


def _xyzm(x: float, y: float, z: float, m: float) -> Coord:
    return {"x": x, "y": y, "z": z, "m": m}


_FINITE = _xyzm(1, 1, 1, 1)
_NAN_Z = _xyzm(1, 1, NAN, 1)
_INF_M = _xyzm(1, 1, 1, INF)
_ONLY_Z_AND_M = _xyzm(NAN, NAN, 1, 1)
_ALL_NAN = _xyzm(NAN, NAN, NAN, NAN)


@pytest.mark.parametrize(
    ("kwargs", "kept"),
    [
        ({}, [_FINITE, _INF_M]),
        ({"include_m": True}, [_FINITE]),
        ({"include_z": False}, [_FINITE, _NAN_Z, _INF_M]),
        ({"include_z": False, "include_m": True}, [_FINITE, _NAN_Z]),
        ({"select_non_finite": False}, [_FINITE, _NAN_Z, _INF_M, _ONLY_Z_AND_M]),
        ({"select_empty": False}, [_FINITE, _INF_M, _ALL_NAN]),
        (
            {"select_empty": False, "include_z": False},
            [_FINITE, _NAN_Z, _INF_M, _ALL_NAN],
        ),
    ],
)
def test_include_z_and_include_m_only_affect_select_non_finite(
    kwargs: dict[str, bool], kept: Line
) -> None:
    """A vertex is only empty with every axis NaN, so `(NaN, NaN, 1, 1)` is not."""
    line = [_FINITE, _NAN_Z, _INF_M, _ONLY_Z_AND_M, _ALL_NAN, _FINITE]

    out = _filter(LineStringXYZM(), line, **kwargs)

    assert out == [_nan_as_str([*kept, _FINITE])]


@pytest.mark.parametrize("action", ["set_empty", "drop"])
def test_a_selected_point_becomes_the_empty_point(action: str) -> None:
    rows = [*_A, *_NOT_FINITE, *_EMPTY_POINT, None]

    assert _filter(PointXY(), *rows, action=action) == [
        *_A,
        _EMPTY,
        _EMPTY,
        _EMPTY,
        None,
    ]


def test_a_selected_point_can_become_null() -> None:
    rows = [*_A, *_NOT_FINITE, *_EMPTY_POINT, None]

    assert _filter(PointXY(), *rows, action="set_null") == [
        *_A,
        None,
        None,
        None,
        None,
    ]


def test_only_the_rows_holding_a_selected_vertex_become_null() -> None:
    out = _filter(
        MultiPolygonXY(),
        [[_SQUARE], [_SQUARE, _line((1, 1), (2, 1), (NAN, NAN), (1, 1))]],
        [[_SQUARE]],
        None,
        action="set_null",
    )

    assert out == [None, [[_SQUARE]], None]


def test_a_multipoint_can_lose_all_of_its_points() -> None:
    assert _filter(MultiPointXY(), _A + _NOT_FINITE, _EMPTY_POINT) == [_A, []]


@pytest.mark.parametrize(
    ("short_lines", "short"),
    [("null", None), ("keep", _A), ("empty", [])],
)
def test_a_line_of_one_vertex_is_too_short(
    short_lines: str, short: Line | None
) -> None:
    out = _filter(
        LineStringXY(),
        _A + _EMPTY_POINT,
        _A,
        _EMPTY_POINT,
        _A + _B,
        short_lines=short_lines,
    )

    assert out == [short, short, [], _A + _B]


@pytest.mark.parametrize(
    ("short_lines", "expected"),
    [("null", None), ("keep", [_A + _B, _A]), ("empty", [_A + _B, []])],
)
def test_a_short_part_of_a_multilinestring(short_lines: str, expected: object) -> None:
    out = _filter(
        MultiLineStringXY(), [_A + _B, _A + _NOT_FINITE], short_lines=short_lines
    )

    assert out == [expected]


_TRIANGLE = _line((0, 0), (4, 0), (4, 4), (0, 0))
_OPEN = _line((0, 0), (4, 0), (4, 4), (0, 4))


@pytest.mark.parametrize(
    ("short_lines", "short", "open_"),
    [
        ("null", None, None),
        ("keep", [_A + _B + _A], [_OPEN]),
        ("empty", [[]], [[]]),
    ],
)
def test_a_ring_that_is_short_or_open_is_too_short(
    short_lines: str, short: object, open_: object
) -> None:
    out = _filter(
        PolygonXY(),
        [_TRIANGLE[:2] + _NOT_FINITE + _TRIANGLE[2:]],
        [_A + _B + _EMPTY_POINT + _A],
        [_A + _B + _A],
        [_EMPTY_POINT + _OPEN + _EMPTY_POINT],
        [_OPEN],
        short_lines=short_lines,
    )

    assert out == [[_TRIANGLE], short, short, open_, open_]


def test_a_ring_that_loses_every_vertex_becomes_empty() -> None:
    assert _filter(PolygonXY(), [_SQUARE, _NOT_FINITE]) == [[_SQUARE, []]]


def test_a_ring_that_is_closed_in_x_and_y_but_not_z_is_too_short() -> None:
    ring = [_xyzm(0, 0, 0, 0), _xyzm(4, 0, 0, 0), _xyzm(4, 4, 0, 0), _xyzm(0, 0, 1, 0)]

    assert _filter(PolygonXYZM(), [ring], short_lines="empty") == [[[]]]


def test_only_the_multipolygon_with_a_short_ring_becomes_null() -> None:
    hole = _line((1, 1), (2, 1), (NAN, NAN), (1, 1))

    out = _filter(MultiPolygonXY(), [[_SQUARE], [_SQUARE, hole]], [[_SQUARE]])

    assert out == [None, [[_SQUARE]]]


def test_a_short_hole_becomes_empty_in_place() -> None:
    hole = _line((1, 1), (2, 1), (NAN, NAN), (1, 1))

    out = _filter(MultiPolygonXY(), [[_SQUARE, hole], [_SQUARE]], short_lines="empty")

    assert out == [[[_SQUARE, []], [_SQUARE]]]


def test_a_short_ring_in_a_sliced_column_misses_its_own_row() -> None:
    hole = _line((1, 1), (2, 1), (NAN, NAN), (1, 1))
    df = _frame(MultiPolygonXY(), [[_SQUARE]], [[_SQUARE, hole]], [[_SQUARE]]).slice(1)

    out = df.select(geo.filter_coordinates("g").ext.storage())

    assert out["g"].to_list() == [None, [[_SQUARE]]]


def test_a_null_row_stays_null() -> None:
    assert _filter(LineStringXY(), None, _A + _NOT_FINITE + _B) == [None, _A + _B]


def test_the_crs_is_carried_over() -> None:
    df = _frame(PolygonXY(crs="EPSG:28992"), [_SQUARE])

    assert df.select(geo.filter_coordinates("g")).schema == df.schema


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [({"action": "remove"}, "action"), ({"short_lines": "drop"}, "short_lines")],
)
def test_an_unknown_option_is_refused(kwargs: dict[str, str], name: str) -> None:
    with pytest.raises(ValueError, match=name):
        geo.filter_coordinates("g", **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=name):
        geo.NonFinite(**kwargs)  # type: ignore[arg-type]
