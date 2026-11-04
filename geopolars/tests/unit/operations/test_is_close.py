"""Whether two geometries are almost the same, coordinate by coordinate."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.exceptions import ComputeError, ShapeError

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PointXYM,
    PointXYZ,
    PolygonXY,
)

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_XYZ = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})
_XYM = pl.Struct({"x": pl.Float64, "y": pl.Float64, "m": pl.Float64})

Coord = dict[str, float]


def _xy(*xy: tuple[float, float]) -> list[Coord]:
    return [{"x": x, "y": y} for x, y in xy]


def _series(name: str, dtype: GeoArrowType, rows: list[object]) -> pl.Series:
    storage = {PointXYZ: _XYZ, PointXYM: _XYM}.get(type(dtype), _XY)
    for _ in range(dtype._nesting):
        storage = pl.List(storage)
    return pl.Series(name, rows, dtype=storage).ext.to(dtype)


def _is_close(
    dtype: GeoArrowType, a: list[object], b: list[object], **kwargs: float | bool
) -> list[bool | None]:
    df = pl.DataFrame([_series("a", dtype, a), _series("b", dtype, b)])
    return df.select(geo.is_close("a", "b", **kwargs))["a"].to_list()


_LINE = _xy((1e6, 0), (2e6, 0), (2e6, 1))
_NUDGED = _xy((1e6, 0), (2e6, 0), (2e6, 1 + 1e-12))
_MOVED = _xy((1e6, 0), (2e6, 0), (2e6, 1.1))

# One case per kind: a geometry, one nudged within the default `rel_tol`, and one moved beyond it.
_KINDS = {
    "point": (PointXY(), _LINE[2], _NUDGED[2], _MOVED[2]),
    "multipoint": (MultiPointXY(), _LINE, _NUDGED, _MOVED),
    "linestring": (LineStringXY(), _LINE, _NUDGED, _MOVED),
    "multilinestring": (
        MultiLineStringXY(),
        [_LINE, _LINE],
        [_LINE, _NUDGED],
        [_LINE, _MOVED],
    ),
    "polygon": (
        PolygonXY(),
        [_LINE + _LINE[:1]],
        [_NUDGED + _LINE[:1]],
        [_MOVED + _LINE[:1]],
    ),
    "multipolygon": (
        MultiPolygonXY(),
        [[_LINE + _LINE[:1]]],
        [[_NUDGED + _LINE[:1]]],
        [[_MOVED + _LINE[:1]]],
    ),
}


@pytest.mark.parametrize("kind", _KINDS)
def test_equal_and_nudged_are_close_moved_is_not(kind: str) -> None:
    dtype, same, nudged, moved = _KINDS[kind]

    assert _is_close(dtype, [same, same, same], [same, nudged, moved]) == [
        True,
        True,
        False,
    ]


@pytest.mark.parametrize("kind", _KINDS)
def test_abs_tol_lets_a_moved_geometry_be_close(kind: str) -> None:
    dtype, same, _, moved = _KINDS[kind]

    assert _is_close(dtype, [same], [moved], abs_tol=0.2) == [True]


@pytest.mark.parametrize("kind", _KINDS)
def test_a_null_geometry_gives_a_null_result(kind: str) -> None:
    dtype, same, _, _ = _KINDS[kind]

    assert _is_close(dtype, [same, None, same], [same, same, None]) == [
        True,
        None,
        None,
    ]


@pytest.mark.parametrize("kind", _KINDS)
def test_a_single_geometry_is_compared_against_every_row(kind: str) -> None:
    dtype, same, nudged, moved = _KINDS[kind]
    many = _series("many", dtype, [same, nudged, moved, None])
    one = _series("one", dtype, [same])
    df = pl.DataFrame(many)

    assert df.select(geo.is_close("many", one))["many"].to_list() == [
        True,
        True,
        False,
        None,
    ]
    assert df.select(geo.is_close(one, "many")).to_series().to_list() == [
        True,
        True,
        False,
        None,
    ]


def test_against_a_single_null_geometry_everything_is_null() -> None:
    many = _series("many", PointXY(), [_LINE[0], _LINE[1]])
    one = _series("one", PointXY(), [None])

    out = pl.DataFrame(many).select(geo.is_close("many", one))

    assert out["many"].to_list() == [None, None]


def test_a_different_number_of_vertices_is_not_close() -> None:
    assert _is_close(LineStringXY(), [_LINE], [_LINE[:2]]) == [False]


def test_the_same_vertices_in_another_order_are_not_close() -> None:
    assert _is_close(LineStringXY(), [_LINE], [_LINE[::-1]]) == [False]


def test_the_same_coordinates_split_differently_are_not_close() -> None:
    a = [_LINE[:1], _LINE[1:]]
    b = [_LINE[:2], _LINE[2:]]

    assert _is_close(MultiLineStringXY(), [a], [b]) == [False]


def test_the_same_rings_in_other_polygons_are_not_close() -> None:
    square = _xy((0, 0), (1, 0), (1, 1), (0, 0))
    a = [[square, square]]
    b = [[square], [square]]

    assert _is_close(MultiPolygonXY(), [a], [b]) == [False]


def test_empty_points_are_only_close_with_nans_equal() -> None:
    empty = {"x": math.nan, "y": math.nan}

    assert _is_close(PointXY(), [empty], [empty]) == [False]
    assert _is_close(PointXY(), [empty], [empty], nans_equal=True) == [True]


def test_m_is_only_compared_when_unique() -> None:
    a = {"x": 1.0, "y": 2.0, "m": 0.0}
    b = a | {"m": 5.0}

    assert _is_close(PointXYM(), [a], [b]) == [True]
    assert _is_close(PointXYM(), [a], [b], unique_m=True) == [False]


def test_z_is_compared_unless_ignored() -> None:
    a = {"x": 1.0, "y": 2.0, "z": 0.0}
    b = a | {"z": 5.0}

    assert _is_close(PointXYZ(), [a], [b]) == [False]
    assert _is_close(PointXYZ(), [a], [b], unique_z=False) == [True]


def test_infinities_are_only_close_to_themselves() -> None:
    inf = {"x": math.inf, "y": 0.0}

    assert _is_close(PointXY(), [inf, inf], [inf, {"x": 1e308, "y": 0.0}]) == [
        True,
        False,
    ]


@pytest.mark.parametrize("value", [-1.0, math.nan])
@pytest.mark.parametrize("name", ["abs_tol", "rel_tol"])
def test_a_tolerance_cannot_be_negative_or_nan(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"`{name}` must be non-negative and not NaN"):
        geo.is_close("a", "b", **{name: value})


def test_different_types_are_refused() -> None:
    df = pl.DataFrame(
        [
            _series("a", PointXY(), [_LINE[0]]),
            _series("b", LineStringXY(), [_LINE]),
        ]
    )

    with pytest.raises(TypeError, match="same type"):
        df.select(geo.is_close("a", "b"))


def test_different_dimensions_are_refused() -> None:
    xyz = pl.Series("b", [{"x": 0.0, "y": 0.0, "z": 0.0}]).ext.to(PointXYZ())
    df = pl.DataFrame([_series("a", PointXY(), [_LINE[0]]), xyz])

    with pytest.raises(TypeError, match="same type"):
        df.select(geo.is_close("a", "b"))


def test_different_crss_are_refused() -> None:
    df = pl.DataFrame(
        [_series("a", PointXY(), [_LINE[0]]), _series("b", PointXY(), [_LINE[0]])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"), geo.set_crs("b", "EPSG:3857"))

    with pytest.raises(ValueError, match="different CRSs"):
        df.select(geo.is_close("a", "b"))


def test_unequal_lengths_are_refused() -> None:
    a = _series("a", PointXY(), [_LINE[0], _LINE[1]])
    b = _series("b", PointXY(), [_LINE[0], _LINE[1], _LINE[2]])

    with pytest.raises((ShapeError, ComputeError)):
        pl.select(geo.is_close(a, b))
