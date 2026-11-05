"""Sorting the parts of a geometry."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    MultiLineStringXY,
    MultiLineStringXYZM,
    MultiPointXY,
    MultiPointXYM,
    MultiPointXYZ,
    MultiPolygonXY,
    PointXY,
    PolygonXY,
)

if TYPE_CHECKING:
    from geopolars.datatypes import GeoArrowType

NAN = float("nan")

Vertices = list[dict[str, float]]


def _vertices(*xy: tuple[float, float]) -> Vertices:
    return [{"x": x, "y": y} for x, y in xy]


def _ring(*xy: tuple[float, float]) -> Vertices:
    """A closed ring through `xy`."""
    return _vertices(*xy, xy[0])


_OUTER = _ring((0, 0), (9, 0), (9, 9), (0, 9))
# These two only differ in their third vertex.
_HOLE_A = _ring((1, 1), (2, 1), (1, 2))
_HOLE_B = _ring((1, 1), (2, 1), (2, 2))
_HOLE_C = _ring((5, 5), (6, 5), (6, 6))


def _column(dtype: GeoArrowType, *rows: object) -> pl.DataFrame:
    series = pl.Series("geometry", list(rows), dtype=dtype._geo_storage)
    return pl.DataFrame(series.ext.to(dtype))


def _sorted(df: pl.DataFrame, *, descending: bool = False) -> list:
    out = df.select(geo.sort_parts("geometry", descending=descending))
    assert out.schema == df.schema
    return out.to_series().ext.storage().to_list()


@pytest.mark.parametrize("dtype", [PointXY(), LineStringXY()])
def test_a_geometry_without_parts_is_returned_as_it_is(dtype: GeoArrowType) -> None:
    row = (
        {"x": 3.0, "y": 1.0}
        if isinstance(dtype, PointXY)
        else _vertices((3, 1), (0, 0))
    )
    df = _column(dtype, row, None)

    assert_frame_equal(df.select(geo.sort_parts("geometry")), df)


@pytest.mark.parametrize("descending", [False, True])
def test_a_multipoint_has_its_points_sorted(descending: bool) -> None:
    points = _vertices((2, 0), (1, 5), (1, 4), (0, 9))
    df = _column(MultiPointXY(), points, None, [])

    expected = _vertices((0, 9), (1, 4), (1, 5), (2, 0))
    if descending:
        expected.reverse()
    assert _sorted(df, descending=descending) == [expected, None, []]


def test_multilinestrings_tied_on_their_first_vertices_are_told_apart_later() -> None:
    lines = [
        _vertices((1, 0), (2, 0), (3, 1)),
        _vertices((1, 0), (2, 0), (3, 0)),
        _vertices((0, 7), (9, 9)),
        _vertices((1, 0), (1, 0)),
    ]
    df = _column(MultiLineStringXY(), lines)

    assert _sorted(df) == [[lines[2], lines[3], lines[1], lines[0]]]


@pytest.mark.parametrize("descending", [False, True])
def test_a_line_that_runs_out_first_is_the_smaller_one(descending: bool) -> None:
    long = _vertices((1, 0), (2, 0))
    short = _vertices((1, 0))
    df = _column(MultiLineStringXY(), [long, [], short])

    expected = [[], short, long]
    if descending:
        expected.reverse()
    assert _sorted(df, descending=descending) == [expected]


@pytest.mark.parametrize("descending", [False, True])
def test_a_polygon_keeps_its_exterior_first(descending: bool) -> None:
    exterior = _ring((5, 5), (9, 5), (9, 9))
    df = _column(PolygonXY(), [exterior, _HOLE_C, _HOLE_A, _HOLE_B], None, [], [_OUTER])

    holes = [_HOLE_A, _HOLE_B, _HOLE_C]
    if descending:
        holes.reverse()
    assert _sorted(df, descending=descending) == [
        [exterior, *holes],
        None,
        [],
        [_OUTER],
    ]


def test_a_multipolygon_has_its_holes_sorted_before_its_polygons() -> None:
    """The two polygons only differ in the order of their holes,
    so once those are sorted, they are equal and keep their order."""
    first = [_OUTER, _HOLE_C, _HOLE_A]
    second = [_OUTER, _HOLE_A, _HOLE_C]
    small = [_ring((0, 0), (1, 0), (1, 1))]
    df = _column(MultiPolygonXY(), [first, small, second], None, [])

    assert _sorted(df) == [[small, second, second], None, []]


@pytest.mark.parametrize("descending", [False, True])
def test_multipolygons_with_the_same_exterior_are_ordered_by_their_holes(
    descending: bool,
) -> None:
    with_b = [_OUTER, _HOLE_B]
    with_a_and_c = [_OUTER, _HOLE_C, _HOLE_A]
    bare = [_OUTER]
    df = _column(MultiPolygonXY(), [with_b, with_a_and_c, bare])

    expected = [bare, [_OUTER, _HOLE_A, _HOLE_C], with_b]
    if descending:
        # The holes are sorted from greatest to smallest too,
        # so `_HOLE_C` now comes before `_HOLE_B`.
        expected = [[_OUTER, _HOLE_C, _HOLE_A], with_b, bare]
    assert _sorted(df, descending=descending) == [expected]


def test_equal_parts_keep_their_order() -> None:
    """`-0.0` and `0.0` are equal, so the sign of the zero shows the order."""
    df = _column(MultiPointXY(), _vertices((0.0, 1), (-0.0, 1), (0.0, 1)))

    for descending in [False, True]:
        out = _sorted(df, descending=descending)[0]
        signs = [math.copysign(1, point["x"]) for point in out]
        assert signs == [1, -1, 1]


def test_nan_is_the_greatest_value() -> None:
    df = _column(MultiPointXY(), _vertices((NAN, 0), (math.inf, 0), (1, NAN), (1, 0)))

    out = [(point["x"], point["y"]) for point in _sorted(df)[0]]

    # NaN does not equal itself, so compare the text.
    assert repr(out) == repr([(1.0, 0.0), (1.0, NAN), (math.inf, 0.0), (NAN, 0.0)])


@pytest.mark.parametrize(
    ("dtype", "axis"),
    [(MultiPointXYZ(), "z"), (MultiPointXYM(), "m")],
)
def test_z_and_m_break_ties(dtype: GeoArrowType, axis: str) -> None:
    points = [{"x": 1.0, "y": 1.0, axis: value} for value in [3.0, 1.0, 2.0]]
    df = _column(dtype, points)

    assert [point[axis] for point in _sorted(df)[0]] == [1.0, 2.0, 3.0]


def test_z_is_compared_before_m() -> None:
    line = [{"x": 0.0, "y": 0.0, "z": 0.0, "m": 0.0}]
    by_m = [{"x": 0.0, "y": 0.0, "z": 0.0, "m": 1.0}]
    by_z = [{"x": 0.0, "y": 0.0, "z": 1.0, "m": -1.0}]
    df = _column(MultiLineStringXYZM(), [by_z, by_m, line])

    assert _sorted(df) == [[line, by_m, by_z]]


def test_the_crs_is_carried_over() -> None:
    df = _column(MultiPointXY(crs="EPSG:28992"), _vertices((1, 0), (0, 0)))

    out = df.select(gpl.col("geometry").geo.sort_parts())

    assert out.schema == df.schema


def test_sorting_twice_changes_nothing() -> None:
    df = _column(MultiPolygonXY(), [[_OUTER, _HOLE_C, _HOLE_A], [_OUTER]])
    once = df.select(geo.sort_parts("geometry"))

    assert_frame_equal(once.select(geo.sort_parts("geometry")), once)


def _random_rows(rng: random.Random, part: Callable[[], object]) -> list[object]:
    return [
        None if rng.random() < 0.1 else [part() for _ in range(rng.randint(0, 6))]
        for _ in range(200)
    ]


def _with_holes_sorted(rings: pl.Expr, *, descending: bool) -> pl.Expr:
    return pl.concat_list(
        rings.list.slice(0, 1), rings.list.slice(1).list.sort(descending=descending)
    )


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [MultiPointXY(), MultiLineStringXY(), PolygonXY(), MultiPolygonXY()],
    ids=["multipoint", "multilinestring", "polygon", "multipolygon"],
)
def test_parts_are_ordered_as_polars_sorts_them(
    dtype: GeoArrowType, descending: bool
) -> None:
    rng = random.Random(42)
    # Few distinct values, so many parts tie on their first vertices.
    values = [0.0, 1.0, NAN]

    def point() -> dict[str, float]:
        return {"x": rng.choice(values), "y": rng.choice(values)}

    def line() -> Vertices:
        return [point() for _ in range(rng.randint(0, 3))]

    def polygon() -> list[Vertices]:
        return [line() for _ in range(rng.randint(0, 4))]

    part = {
        MultiPointXY: point,
        MultiLineStringXY: line,
        PolygonXY: line,
        MultiPolygonXY: polygon,
    }[type(dtype)]
    rows = _random_rows(rng, part)
    df = _column(dtype, *rows)
    storage = pl.col("geometry").ext.storage()

    if isinstance(dtype, PolygonXY):
        expected = _with_holes_sorted(storage, descending=descending)
    elif isinstance(dtype, MultiPolygonXY):
        expected = storage.list.eval(
            _with_holes_sorted(pl.element(), descending=descending)
        ).list.sort(descending=descending)
    else:
        expected = storage.list.sort(descending=descending)

    out = df.select(geo.sort_parts("geometry", descending=descending).ext.storage())
    assert_series_equal(out.to_series(), df.select(expected).to_series())
