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
    LineStringXYM,
    LineStringXYZ,
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


def _sorted(df: pl.DataFrame, **kwargs: object) -> list:
    out = df.select(geo.sort_parts("geometry", **kwargs))  # type: ignore[arg-type]
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


def _sorted_by_polars(dtype: GeoArrowType, *, descending: bool) -> pl.Expr:
    """The storage of the `geometry` column, with its parts sorted by Polars."""
    storage = pl.col("geometry").ext.storage()
    if isinstance(dtype, LineStringXY):
        return storage
    if isinstance(dtype, PolygonXY):
        return _with_holes_sorted(storage, descending=descending)
    if isinstance(dtype, MultiPolygonXY):
        return storage.list.eval(
            _with_holes_sorted(pl.element(), descending=descending)
        ).list.sort(descending=descending)
    return storage.list.sort(descending=descending)


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

    out = df.select(geo.sort_parts("geometry", descending=descending).ext.storage())
    expected = df.select(_sorted_by_polars(dtype, descending=descending))
    assert_series_equal(out.to_series(), expected.to_series())


# Rotating rings, with `rings="rotate"`.


def _rotated(ring: Vertices) -> Vertices:
    """`ring` begun at its smallest rotation, closed again by its new first vertex."""
    distinct = ring[:-1]
    if len(distinct) < 2:
        return ring
    rotations = [distinct[k:] + distinct[:k] for k in range(len(distinct))]
    # `min` keeps the earliest of equal rotations.
    k = min(
        range(len(rotations)),
        key=lambda k: [tuple(vertex.values()) for vertex in rotations[k]],
    )
    return ring if k == 0 else [*rotations[k], rotations[k][0]]


def _rotated_rows(dtype: GeoArrowType, rows: list[object]) -> list[object]:
    def rotated(row: object, depth: int) -> object:
        if row is None:
            return None
        if depth == 0:
            return _rotated(row)  # type: ignore[arg-type]
        return [rotated(part, depth - 1) for part in row]  # type: ignore[attr-defined]

    depth = {LineStringXY: 0, MultiLineStringXY: 1, PolygonXY: 1, MultiPolygonXY: 2}
    return [rotated(row, depth[type(dtype)]) for row in rows]


def _rotated_out(df: pl.DataFrame, **kwargs: object) -> list:
    return _sorted(df, rings="rotate", **kwargs)


def test_a_linestring_starts_at_its_smallest_vertex() -> None:
    ring = _ring((2, 0), (0, 1), (0, 0), (1, 5))
    canonical = _ring((0, 0), (1, 5), (2, 0), (0, 1))
    df = _column(LineStringXY(), ring, canonical, None, [])

    assert _rotated_out(df) == [canonical, canonical, None, []]


def test_a_ring_through_its_smallest_vertex_twice_is_decided_by_the_next() -> None:
    df = _column(LineStringXY(), _ring((0, 0), (5, 0), (0, 0), (1, 0)))

    assert _rotated_out(df) == [_ring((0, 0), (1, 0), (0, 0), (5, 0))]


def test_a_ring_with_equally_small_starts_keeps_the_earliest() -> None:
    df = _column(LineStringXY(), _ring((1, 1), (0, 0), (1, 1), (0, 0)))

    assert _rotated_out(df) == [_ring((0, 0), (1, 1), (0, 0), (1, 1))]


def test_a_ring_keeps_its_direction() -> None:
    ring = _ring((1, 0), (1, 1), (0, 0))
    df = _column(LineStringXY(), ring, ring[::-1])

    assert _rotated_out(df) == [
        _ring((0, 0), (1, 0), (1, 1)),
        _ring((0, 0), (1, 1), (1, 0)),
    ]


def test_a_rotated_ring_is_closed_by_its_new_first_vertex() -> None:
    """The old closing vertex only matched the first in x and y, so its m is lost."""
    xym = [(1, 0, 1), (0, 0, 2), (1, 1, 3), (1, 0, 9)]
    ring = [{"x": x, "y": y, "m": m} for x, y, m in xym]
    df = _column(LineStringXYM(), ring)

    assert _rotated_out(df) == [[ring[1], ring[2], ring[0], ring[1]]]


def test_z_decides_between_starts_equal_in_x_and_y() -> None:
    xyz = [(0, 0, 5), (1, 0, 0), (0, 0, 1), (0, 1, 0)]
    vertices = [{"x": x, "y": y, "z": z} for x, y, z in xyz]
    df = _column(LineStringXYZ(), [*vertices, vertices[0]])

    assert _rotated_out(df) == [[*vertices[2:], *vertices[:2], vertices[2]]]


def test_a_polygon_has_every_ring_rotated_before_its_holes_are_sorted() -> None:
    exterior = _ring((9, 9), (0, 0), (9, 0))
    hole_at_5 = _ring((6, 5), (6, 6), (5, 5))
    # Sorts after `hole_at_5` as it is, but before it once rotated.
    hole_at_1 = _ring((7, 0), (1, 1), (2, 2))
    df = _column(PolygonXY(), [exterior, hole_at_1, hole_at_5])

    assert _sorted(df) == [[exterior, hole_at_5, hole_at_1]]
    assert _rotated_out(df) == [
        [_rotated(exterior), _rotated(hole_at_1), _rotated(hole_at_5)]
    ]


def test_multipolygons_that_only_differ_in_their_starts_tie() -> None:
    ring = _ring((0, 0), (1, 0), (1, 1))
    same_ring_rotated = _ring((1, 0), (1, 1), (0, 0))
    df = _column(MultiPolygonXY(), [[same_ring_rotated], [ring]])

    assert _sorted(df) == [[[ring], [same_ring_rotated]]]
    assert _rotated_out(df) == [[[ring], [ring]]]


@pytest.mark.parametrize(
    "line",
    [
        _vertices((0, 0), (1, 0), (1, 1), (0, 1)),
        _vertices((0, 0), (1, 0), (0, 0)),
    ],
    ids=["open", "too short"],
)
def test_a_line_that_is_not_a_ring_raises(line: Vertices) -> None:
    ring = _ring((1, 0), (1, 1), (0, 0))
    lines = _column(LineStringXY(), ring, line)
    multi = _column(MultiLineStringXY(), [ring], [ring, line])

    for df in [lines, multi]:
        with pytest.raises(pl.exceptions.ComputeError, match="expects rings"):
            _rotated_out(df)
        assert _rotated_out(df, ignore_per_row_errors=True)[1] is None


def test_a_line_that_is_not_a_ring_only_nulls_its_own_row() -> None:
    ring = _ring((1, 0), (1, 1), (0, 0))
    df = _column(LineStringXY(), ring, _vertices((0, 0), (1, 0)), None)

    assert _rotated_out(df, ignore_per_row_errors=True) == [_rotated(ring), None, None]


def test_empty_lines_are_kept_and_sort_first() -> None:
    ring = _ring((1, 0), (1, 1), (0, 0))
    df = _column(MultiLineStringXY(), [ring, []], [], None)

    assert _rotated_out(df) == [[[], _rotated(ring)], [], None]


def test_descending_does_not_change_the_starts() -> None:
    low = _ring((1, 0), (1, 1), (0, 0))
    high = _ring((5, 5), (4, 4), (5, 4))
    df = _column(MultiLineStringXY(), [low, high])

    assert _rotated_out(df, descending=True) == [[_rotated(high), _rotated(low)]]


def test_points_ignore_rotating_rings() -> None:
    df = _column(MultiPointXY(), _vertices((2, 0), (1, 5)))

    assert _rotated_out(df) == _sorted(df)


def test_an_unknown_rings_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="`rings` must be one of"):
        geo.sort_parts("geometry", rings="spin")  # type: ignore[arg-type]


def test_the_namespace_passes_on_rotating_rings() -> None:
    df = _column(LineStringXY(crs="EPSG:28992"), _ring((1, 0), (1, 1), (0, 0)))

    out = df.select(gpl.col("geometry").geo.sort_parts(rings="rotate"))

    assert out.schema == df.schema
    assert out.to_series().ext.storage().to_list() == [_ring((0, 0), (1, 0), (1, 1))]


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [LineStringXY(), MultiLineStringXY(), PolygonXY(), MultiPolygonXY()],
    ids=["linestring", "multilinestring", "polygon", "multipolygon"],
)
def test_rotated_parts_are_ordered_as_polars_sorts_them(
    dtype: GeoArrowType, descending: bool
) -> None:
    rng = random.Random(7)

    def ring() -> Vertices:
        if rng.random() < 0.1:
            return []
        # Few distinct values, so many rings share vertices and tie on their starts.
        values = [0.0, 1.0]
        vertices = [
            {"x": rng.choice(values), "y": rng.choice(values)}
            for _ in range(rng.randint(3, 6))
        ]
        return [*vertices, vertices[0]]

    def polygon() -> list[Vertices]:
        return [ring() for _ in range(rng.randint(0, 4))]

    if isinstance(dtype, LineStringXY):
        rows: list[object] = [
            None if rng.random() < 0.1 else ring() for _ in range(200)
        ]
    else:
        part = polygon if isinstance(dtype, MultiPolygonXY) else ring
        rows = _random_rows(rng, part)

    out = _column(dtype, *rows).select(
        geo.sort_parts("geometry", descending=descending, rings="rotate").ext.storage()
    )
    expected = _column(dtype, *_rotated_rows(dtype, rows)).select(
        _sorted_by_polars(dtype, descending=descending)
    )
    assert_series_equal(out.to_series(), expected.to_series())
