"""Removing the points that coincide with the one before them."""

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

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_XYZM = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "m": pl.Float64})

Coord = dict[str, float]
Line = list[Coord]


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


_SQUARE = _line((0, 0), (4, 0), (4, 4), (0, 4), (0, 0))
_REPEATED = _line((0, 0), (1, 1), (1, 1), (2, 0))
_REMOVED = _line((0, 0), (1, 1), (2, 0))


def _storage(dtype: GeoArrowType) -> pl.DataType:
    """The storage of `dtype`, with every coordinate as `_XY` or `_XYZM`."""
    coords = _XYZM if isinstance(dtype, LineStringXYZM) else _XY
    for _ in range(dtype._nesting):
        coords = pl.List(coords)
    return coords


def _frame(dtype: GeoArrowType, *rows: object) -> pl.DataFrame:
    """`rows`, relabelled as `dtype` without any checks."""
    storage = pl.Series("g", list(rows), dtype=_storage(dtype))
    return pl.DataFrame(storage.ext.to(dtype))


def _remove(dtype: GeoArrowType, *rows: object, **kwargs: object) -> list:
    out = _frame(dtype, *rows).select(geo.remove_repeats_in_succession("g", **kwargs))
    return out.select(pl.col("g").ext.storage())["g"].to_list()


def test_a_vertex_at_the_same_place_as_the_one_before_is_removed() -> None:
    assert _remove(LineStringXY(), _REPEATED) == [_REMOVED]


@pytest.mark.parametrize(
    ("tolerance", "removed"),
    [({}, False), ({"rel_tol": 1e-9}, True), ({"abs_tol": 1e-9}, True)],
    ids=["exact", "rel_tol", "abs_tol"],
)
def test_vertices_that_are_almost_the_same_need_a_tolerance(
    tolerance: dict[str, float], removed: bool
) -> None:
    line = _line((0, 0), (1, 1), (1 + 1e-12, 1), (2, 0))
    expected = _line((0, 0), (1, 1), (2, 0)) if removed else line

    assert _remove(LineStringXY(), line, **tolerance) == [expected]


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
@pytest.mark.parametrize("name", ["abs_tol", "rel_tol"])
def test_a_tolerance_has_to_be_finite_and_non_negative(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"`{name}` must be finite and non-negative"):
        geo.remove_repeats_in_succession("g", **{name: value})


def test_a_vertex_is_compared_with_the_vertex_kept_before_it() -> None:
    """Each step is within the tolerance, but the drift as a whole is not."""
    line = _line((0, 0), (0.6, 0), (1.2, 0), (1.8, 0), (5, 0))

    assert _remove(LineStringXY(), line, abs_tol=1) == [_line((0, 0), (1.2, 0), (5, 0))]


def test_the_last_vertex_is_kept_in_place_of_the_one_before() -> None:
    line = _line((0, 0), (2, 0), (2.5, 0))

    assert _remove(LineStringXY(), line, abs_tol=1) == [_line((0, 0), (2.5, 0))]


def test_a_vertex_repeated_further_on_is_kept() -> None:
    line = _line((0, 0), (1, 1), (0, 0))

    assert _remove(LineStringXY(), line) == [line]


def test_a_ring_stays_closed() -> None:
    ring = _line((0, 0), (4, 0), (4, 0), (4, 4), (0, 4), (0, 0), (0, 0))

    assert _remove(PolygonXY(), [ring]) == [[_SQUARE]]


def test_a_polygon_with_a_collapsed_ring_becomes_null() -> None:
    ring = _line((0, 0), (1e-12, 0), (1, 1), (0, 0))

    assert _remove(PolygonXY(), [ring], [_SQUARE], abs_tol=1e-9) == [None, [_SQUARE]]


def test_a_multipolygon_with_a_collapsed_ring_becomes_null() -> None:
    hole = _line((1, 1), (1, 1), (2, 2), (1, 1))

    assert _remove(MultiPolygonXY(), [[_SQUARE], [_SQUARE, hole]], [[_SQUARE]]) == [
        None,
        [[_SQUARE]],
    ]


def test_a_collapsed_ring_in_a_sliced_column_misses_its_own_row() -> None:
    hole = _line((1, 1), (1, 1), (2, 2), (1, 1))
    df = _frame(MultiPolygonXY(), [[_SQUARE]], [[_SQUARE, hole]], [[_SQUARE]]).slice(1)

    out = df.select(geo.remove_repeats_in_succession("g").ext.storage())

    assert out["g"].to_list() == [None, [[_SQUARE]]]


_A = _line((1, 1))


@pytest.mark.parametrize(
    ("collapsed_lines", "too_short", "collapsed", "single"),
    [
        ("keep_ends", "null", _A + _A, None),
        ("keep_ends", "keep", _A + _A, _A),
        ("keep_ends", "empty", _A + _A, []),
        ("single_vertex", "null", None, None),
        ("single_vertex", "keep", _A, _A),
        ("single_vertex", "empty", [], []),
    ],
)
def test_what_a_collapsed_or_short_line_keeps(
    collapsed_lines: str,
    too_short: str,
    collapsed: Line | None,
    single: Line | None,
) -> None:
    out = _remove(
        LineStringXY(),
        _A + _A + _A,
        _A,
        collapsed_lines=collapsed_lines,
        too_short=too_short,
    )

    assert out == [collapsed, single]


@pytest.mark.parametrize(
    ("too_short", "expected"),
    [("null", None), ("keep", [_REMOVED, _A]), ("empty", [_REMOVED, []])],
)
def test_a_short_part_of_a_multilinestring(too_short: str, expected: object) -> None:
    out = _remove(
        MultiLineStringXY(),
        [_REPEATED, _A + _A],
        collapsed_lines="single_vertex",
        too_short=too_short,
    )

    assert out == [expected]


@pytest.mark.parametrize("too_short", ["null", "keep", "empty"])
def test_a_multipoint_of_one_point_repeated_keeps_one(too_short: str) -> None:
    out = _remove(MultiPointXY(), _A + _A + _A, too_short=too_short)

    assert out == [_A]


_B = _line((2, 2))


@pytest.mark.parametrize("collapsed_lines", ["keep_ends", "single_vertex"])
@pytest.mark.parametrize(
    ("too_short", "at_one_place", "back_and_forth"),
    [
        ("null", None, None),
        ("keep", [_A + _A], [_A + _B + _A]),
        ("empty", [[]], [[]]),
    ],
)
def test_what_a_short_ring_keeps(
    collapsed_lines: str, too_short: str, at_one_place: object, back_and_forth: object
) -> None:
    out = _remove(
        PolygonXY(),
        [_A + _A + _A + _A],
        [_A + _B + _B + _A],
        [_A + _A + _B + _A],
        collapsed_lines=collapsed_lines,
        too_short=too_short,
    )

    assert out == [at_one_place, back_and_forth, back_and_forth]


def test_a_ring_that_was_short_already() -> None:
    assert _remove(PolygonXY(), [_A + _B + _A], too_short="empty") == [[[]]]


def test_a_short_exterior_or_hole_becomes_empty_in_place() -> None:
    hole = _line((1, 1), (2, 1), (2, 2), (1, 1))

    out = _remove(
        MultiPolygonXY(),
        [[_A + _A + _A + _A, hole], [_SQUARE, _A + _B + _A]],
        too_short="empty",
    )

    assert out == [[[[], hole], [_SQUARE, []]]]


def test_an_unknown_collapsed_lines_is_refused() -> None:
    with pytest.raises(ValueError, match="collapsed_lines"):
        geo.remove_repeats_in_succession("g", collapsed_lines="first")  # type: ignore[arg-type]


def test_an_unknown_too_short_is_refused() -> None:
    with pytest.raises(ValueError, match="too_short"):
        geo.remove_repeats_in_succession("g", too_short="drop")  # type: ignore[arg-type]


def test_only_a_multipoint_point_in_a_row_is_removed() -> None:
    a, b = {"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}

    assert _remove(MultiPointXY(), [a, a, b, a]) == [[a, b, a]]


def test_z_is_compared_and_m_is_not() -> None:
    """Of a run, the first vertex is kept, with its m."""
    a = {"x": 1.0, "y": 1.0, "z": 0.0, "m": 0.0}
    b = a | {"m": 1.0}
    c = a | {"z": 1.0}

    assert _remove(LineStringXYZM(), [a, b, c]) == [[a, c]]


_P = {"x": 1.0, "y": 1.0, "z": 0.0, "m": 0.0}
_OTHER_M = _P | {"m": 1.0}
_OTHER_M_AND_Z = _OTHER_M | {"z": 1.0}
_END = {"x": 2.0, "y": 2.0, "z": 0.0, "m": 0.0}


@pytest.mark.parametrize(
    ("unique_z", "unique_m", "kept"),
    [
        (True, False, [_P, _OTHER_M_AND_Z, _END]),
        (False, False, [_P, _END]),
        (True, True, [_P, _OTHER_M, _OTHER_M_AND_Z, _END]),
        (False, True, [_P, _OTHER_M, _END]),
    ],
)
def test_unique_z_and_unique_m_choose_the_axes_compared(
    unique_z: bool, unique_m: bool, kept: Line
) -> None:
    line = [_P, _OTHER_M, _OTHER_M_AND_Z, _END]

    out = _remove(LineStringXYZM(), line, unique_z=unique_z, unique_m=unique_m)

    assert out == [kept]


@pytest.mark.parametrize(("abs_tol", "removed"), [(None, False), (1e-9, True)])
def test_a_compared_m_is_within_the_tolerance(
    abs_tol: float | None, removed: bool
) -> None:
    almost = _P | {"m": 1e-12}
    expected = [_P, _END] if removed else [_P, almost, _END]

    out = _remove(LineStringXYZM(), [_P, almost, _END], abs_tol=abs_tol, unique_m=True)

    assert out == [expected]


def test_unique_z_and_unique_m_are_ignored_without_those_axes() -> None:
    out = _remove(LineStringXY(), _REPEATED, unique_z=True, unique_m=True)

    assert out == [_REMOVED]


@pytest.mark.parametrize(
    ("dtype", "row", "expected"),
    [
        (
            MultiLineStringXY(),
            [_line((5, 5), (6, 6)), _REPEATED],
            [_line((5, 5), (6, 6)), _REMOVED],
        ),
        (
            MultiPolygonXY(),
            [[_SQUARE], [_REPEATED + _REPEATED[:1]]],
            [[_SQUARE], [_REMOVED + _REMOVED[:1]]],
        ),
    ],
    ids=["multilinestring", "multipolygon"],
)
def test_every_part_of_a_multi_geometry_is_handled(
    dtype: GeoArrowType, row: object, expected: object
) -> None:
    assert _remove(dtype, row) == [expected]


def test_a_point_is_returned_as_is() -> None:
    point = {"x": 0.0, "y": 0.0}

    assert _remove(PointXY(), point, None) == [point, None]


def test_null_and_empty_geometries_stay_so() -> None:
    assert _remove(LineStringXY(), None, [], _REPEATED) == [None, [], _REMOVED]


def test_a_sliced_column() -> None:
    df = _frame(MultiLineStringXY(), [_REPEATED], [_SQUARE, _REPEATED]).slice(1)

    out = df.select(geo.remove_repeats_in_succession("g").ext.storage())

    assert out["g"].to_list() == [[_SQUARE, _REMOVED]]


@pytest.mark.parametrize(
    ("dtype", "row"),
    [
        (LineStringXY(), _REPEATED),
        (PolygonXY(), [_REPEATED + _REPEATED[:1]]),
        (MultiLineStringXY(), [_REPEATED, _REPEATED]),
        (MultiPolygonXY(), [[_REPEATED + _REPEATED[:1]]]),
    ],
    ids=["linestring", "polygon", "multilinestring", "multipolygon"],
)
def test_no_repeated_points_are_left(dtype: GeoArrowType, row: object) -> None:
    out = _frame(dtype, row).select(
        geo.remove_repeats_in_succession("g").geo.has_repeats_in_succession()
    )

    assert out["g"].to_list() == [False]


def test_the_crs_is_carried_over() -> None:
    df = _frame(LineStringXY(crs="EPSG:28992"), _REPEATED)

    out = df.select(geo.remove_repeats_in_succession("g"))

    assert out.schema == df.schema


def test_namespace_matches_the_functional_api() -> None:
    df = _frame(LineStringXY(), _REPEATED, _SQUARE)

    assert_frame_equal(
        df.select(gpl.col("g").geo.remove_repeats_in_succession(abs_tol=0.1)),
        df.select(geo.remove_repeats_in_succession("g", abs_tol=0.1)),
    )
