"""Whether a geometry has two points that coincide."""

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


def _has_repeats_in_succession(
    dtype: GeoArrowType, *rows: object, **kwargs: float | bool | None
) -> list[bool | None]:
    out = _frame(dtype, *rows).select(geo.has_repeats_in_succession("g", **kwargs))
    return out["g"].to_list()


def test_two_vertices_in_a_row_at_the_same_place_are_repeated() -> None:
    assert _has_repeats_in_succession(LineStringXY(), _REPEATED) == [True]


@pytest.mark.parametrize(
    ("tolerance", "repeated"),
    [({}, False), ({"rel_tol": 1e-9}, True), ({"abs_tol": 1e-9}, True)],
    ids=["exact", "rel_tol", "abs_tol"],
)
def test_vertices_in_a_row_that_are_almost_the_same_need_a_tolerance(
    tolerance: dict[str, float], repeated: bool
) -> None:
    line = _line((0, 0), (1, 1), (1 + 1e-12, 1), (2, 0))

    assert _has_repeats_in_succession(LineStringXY(), line, **tolerance) == [repeated]


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
@pytest.mark.parametrize("name", ["abs_tol", "rel_tol"])
def test_a_tolerance_has_to_be_finite_and_non_negative(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"`{name}` must be finite and non-negative"):
        geo.has_repeats_in_succession("g", **{name: value})


def test_vertices_in_a_row_that_differ_only_in_m_are_repeated_unless_m_is_unique() -> (
    None
):
    """m is a measure along the line, not a place on it."""
    a = {"x": 1.0, "y": 1.0, "z": 0.0, "m": 0.0}
    b = a | {"m": 1.0}

    assert _has_repeats_in_succession(LineStringXYZM(), [a, b]) == [True]
    assert _has_repeats_in_succession(LineStringXYZM(), [a, b], unique_m=True) == [
        False
    ]


def test_vertices_in_a_row_that_differ_in_z_are_not_repeated_unless_z_is_ignored() -> (
    None
):
    a = {"x": 1.0, "y": 1.0, "z": 0.0, "m": 0.0}
    b = a | {"z": 1.0}

    assert _has_repeats_in_succession(LineStringXYZM(), [a, b]) == [False]
    assert _has_repeats_in_succession(LineStringXYZM(), [a, b], unique_z=False) == [
        True
    ]


def test_a_vertex_repeated_further_on_is_not_repeated() -> None:
    line = _line((0, 0), (1, 1), (0, 0))

    assert _has_repeats_in_succession(LineStringXY(), line) == [False]


def test_the_ends_of_a_ring_are_not_in_a_row() -> None:
    assert _has_repeats_in_succession(PolygonXY(), [_SQUARE]) == [False]


def test_a_ring_with_vertices_in_a_row_at_the_same_place_is_repeated() -> None:
    ring = _line((0, 0), (4, 0), (4, 0), (4, 4), (0, 0))

    assert _has_repeats_in_succession(PolygonXY(), [ring]) == [True]


def test_a_multipoint_with_two_points_in_a_row_at_the_same_place_is_repeated() -> None:
    """Only two points in a row count, as for a linestring."""
    a, b = {"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}

    assert _has_repeats_in_succession(MultiPointXY(), [a, a, b], [a, b, a]) == [
        True,
        False,
    ]


def test_a_point_is_never_repeated() -> None:
    assert _has_repeats_in_succession(PointXY(), {"x": 0.0, "y": 0.0}) == [False]


@pytest.mark.parametrize(
    ("dtype", "repeated", "other"),
    [
        (MultiLineStringXY(), [_line((5, 5), (6, 6)), _REPEATED], [_REPEATED[:2]]),
        (MultiPolygonXY(), [[_SQUARE], [_REPEATED + _REPEATED[:1]]], [[_SQUARE]]),
    ],
    ids=["multilinestring", "multipolygon"],
)
def test_a_multi_geometry_is_repeated_when_a_part_is(
    dtype: GeoArrowType, repeated: object, other: object
) -> None:
    assert _has_repeats_in_succession(dtype, repeated, other) == [True, False]


@pytest.mark.parametrize(
    "dtype", [PointXY(), LineStringXY(), PolygonXY(), MultiPolygonXY()]
)
def test_a_null_geometry_gives_a_null_result(dtype: GeoArrowType) -> None:
    assert _has_repeats_in_succession(dtype, None) == [None]


def test_namespace_matches_the_functional_api() -> None:
    df = _frame(LineStringXY(), _REPEATED, _SQUARE)

    assert_frame_equal(
        df.select(gpl.col("g").geo.has_repeats_in_succession(abs_tol=1.0)),
        df.select(geo.has_repeats_in_succession("g", abs_tol=1.0)),
    )
