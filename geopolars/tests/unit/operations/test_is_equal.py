"""Whether two geometries are equal, coordinate by coordinate."""

from __future__ import annotations

import math
from typing import Any

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
    PointXYZM,
    PolygonXY,
)

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_XYZ = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})
_XYM = pl.Struct({"x": pl.Float64, "y": pl.Float64, "m": pl.Float64})
_XYZM = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "m": pl.Float64})

Coord = dict[str, float]


def _xy(*xy: tuple[float, float]) -> list[Coord]:
    return [{"x": x, "y": y} for x, y in xy]


def _series(name: str, dtype: GeoArrowType, rows: list[object]) -> pl.Series:
    storage = {PointXYZ: _XYZ, PointXYM: _XYM, PointXYZM: _XYZM}.get(type(dtype), _XY)
    for _ in range(dtype._nesting):
        storage = pl.List(storage)
    return pl.Series(name, rows, dtype=storage).ext.to(dtype)


def _is_equal(
    dtype: GeoArrowType, a: list[object], b: list[object], **kwargs: float | bool
) -> list[bool | None]:
    df = pl.DataFrame([_series("a", dtype, a), _series("b", dtype, b)])
    return df.select(geo.is_equal("a", "b", **kwargs))["a"].to_list()


def _is_close(
    dtype: GeoArrowType, a: list[object], b: list[object], **kwargs: float | bool
) -> list[bool | None]:
    """`is_equal` with `rel_tol=1e-9` unless given."""
    return _is_equal(dtype, a, b, **({"rel_tol": 1e-9} | kwargs))


_LINE = _xy((1e6, 0), (2e6, 0), (2e6, 1))
_NUDGED = _xy((1e6, 0), (2e6, 0), (2e6, 1 + 1e-12))
_MOVED = _xy((1e6, 0), (2e6, 0), (2e6, 1.1))

# One case per kind: a geometry, one nudged within `rel_tol=1e-9`, and one moved beyond it.
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
def test_without_a_tolerance_only_the_same_geometry_is_equal(kind: str) -> None:
    dtype, same, nudged, _ = _KINDS[kind]
    df = pl.DataFrame(
        [_series("a", dtype, [same, same]), _series("b", dtype, [same, nudged])]
    )

    assert df.select(geo.is_equal("a", "b"))["a"].to_list() == [True, False]


@pytest.mark.parametrize(
    "tolerance",
    [{}, {"abs_tol": 1e-9}, {"rel_tol": 1e-9}, {"abs_tol": 1e-9, "rel_tol": 1e-9}],
    ids=["exact", "abs_tol", "rel_tol", "both"],
)
@pytest.mark.parametrize("kind", _KINDS)
def test_either_tolerance_on_its_own_lets_a_nudged_geometry_be_equal(
    kind: str, tolerance: dict[str, float]
) -> None:
    dtype, same, nudged, moved = _KINDS[kind]

    out = _is_equal(dtype, [same, same, same], [same, nudged, moved], **tolerance)
    assert out == [True, bool(tolerance), False]


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

    close = {"rel_tol": 1e-9}

    assert df.select(geo.is_equal("many", one, **close))["many"].to_list() == [
        True,
        True,
        False,
        None,
    ]
    assert df.select(geo.is_equal(one, "many", **close)).to_series().to_list() == [
        True,
        True,
        False,
        None,
    ]


def test_against_a_single_null_geometry_everything_is_null() -> None:
    many = _series("many", PointXY(), [_LINE[0], _LINE[1]])
    one = _series("one", PointXY(), [None])

    out = pl.DataFrame(many).select(geo.is_equal("many", one))

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


def test_empty_points_are_equal_unless_nans_equal_is_false() -> None:
    empty = {"x": math.nan, "y": math.nan}

    assert _is_equal(PointXY(), [empty], [empty]) == [True]
    assert _is_equal(PointXY(), [empty], [empty], nans_equal=False) == [False]


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


def _point(x: float, y: float) -> Coord:
    return {"x": x, "y": y}


@pytest.mark.parametrize(
    ("a", "b", "tolerance", "equal"),
    [
        ((1e6, 0), (1e6 + 1e-4, 0), {"rel_tol": 1e-9}, True),
        ((1e6, 0), (1e6 + 1e-4, 0), {"abs_tol": 1e-9}, False),
        ((0, 0), (1e-12, 0), {"rel_tol": 1e-9}, False),
        ((0, 0), (1e-12, 0), {"abs_tol": 1e-9, "rel_tol": 1e-9}, True),
        ((1e6, 0), (1e6 + 1e-4, 0), {"abs_tol": 1e-9, "rel_tol": 1e-9}, True),
    ],
    ids=[
        "rel_tol-large",
        "abs_tol-large",
        "rel_tol-near-zero",
        "both-near-zero",
        "both-large",
    ],
)
def test_rel_tol_scales_with_the_values_and_each_tolerance_counts_beside_the_other(
    a: tuple[float, float],
    b: tuple[float, float],
    tolerance: dict[str, float],
    equal: bool,
) -> None:
    assert _is_equal(PointXY(), [_point(*a)], [_point(*b)], **tolerance) == [equal]


@pytest.mark.parametrize(("offset", "equal"), [(1e-12, True), (1.0, False)])
@pytest.mark.parametrize("axis", ["z", "m"])
@pytest.mark.parametrize("tolerance", [{"abs_tol": 1e-9}, {"rel_tol": 1e-9}])
def test_a_compared_z_or_m_has_to_be_within_the_tolerance(
    tolerance: dict[str, float], axis: str, offset: float, equal: bool
) -> None:
    a = {"x": 1.0, "y": 1.0, "z": 1.0, "m": 1.0}
    b = a | {axis: 1.0 + offset}

    out = _is_equal(PointXYZM(), [a], [b], unique_m=True, **tolerance)
    assert out == [equal]


_INFINITE_TOLERANCES = [{"abs_tol": math.inf}, {"rel_tol": math.inf}]

_NON_FINITE_PAIRS = [
    ((0, 0), (math.inf, 0), False),
    ((math.inf, 0), (math.inf, 0), True),
    ((math.inf, 0), (-math.inf, 0), False),
    ((math.nan, 0), (math.nan, 0), False),
]
_NON_FINITE_IDS = ["finite-inf", "inf-inf", "inf-neg-inf", "nan-nan"]


@pytest.mark.parametrize(("a", "b", "equal"), _NON_FINITE_PAIRS, ids=_NON_FINITE_IDS)
@pytest.mark.parametrize(
    "tolerance",
    [{}, {"abs_tol": 1e-9}, {"rel_tol": 1e-9}],
    ids=["exact", "abs_tol", "rel_tol"],
)
def test_an_infinity_is_only_equal_to_itself_and_nan_to_nothing_without_nans_equal(
    tolerance: dict[str, float],
    a: tuple[float, float],
    b: tuple[float, float],
    equal: bool,
) -> None:
    out = _is_equal(
        PointXY(), [_point(*a)], [_point(*b)], nans_equal=False, **tolerance
    )
    assert out == [equal]


@pytest.mark.parametrize(
    "tolerance",
    [{}, {"abs_tol": 1e-9}, {"rel_tol": 1e-9}],
    ids=["exact", "abs_tol", "rel_tol"],
)
def test_a_nan_is_equal_to_a_nan_by_default_whatever_the_tolerance(
    tolerance: dict[str, float],
) -> None:
    a = [_point(math.nan, 0), _point(math.nan, 0)]
    b = [_point(math.nan, 0), _point(0, 0)]

    assert _is_equal(PointXY(), a, b, **tolerance) == [True, False]


@pytest.mark.parametrize(
    ("a", "b", "equal"),
    [((-1e308, 0), (1e308, 0), True), *_NON_FINITE_PAIRS],
    ids=["finite", *_NON_FINITE_IDS],
)
@pytest.mark.parametrize("tolerance", _INFINITE_TOLERANCES)
def test_an_infinite_tolerance_makes_finite_values_equal_and_an_infinity_only_itself(
    tolerance: dict[str, float],
    a: tuple[float, float],
    b: tuple[float, float],
    equal: bool,
) -> None:
    out = _is_equal(
        PointXY(), [_point(*a)], [_point(*b)], nans_equal=False, **tolerance
    )
    assert out == [equal]


def _chunked(series: list[pl.Series]) -> pl.Series:
    out = pl.concat(series, rechunk=False)
    assert out.n_chunks() == len(series)
    return out


def test_a_chunked_column_is_compared_against_a_single_geometry() -> None:
    dtype, same, nudged, moved = _KINDS["linestring"]
    many = _chunked(
        [
            _series("many", dtype, [same, nudged]),
            _series("many", dtype, [None, moved, same]),
        ]
    )
    one = _series("one", dtype, [same])
    df = pl.DataFrame(many)
    expected = [True, False, None, False, True]

    assert df.select(geo.is_equal("many", one))["many"].to_list() == expected
    assert df.select(geo.is_equal(one, "many")).to_series().to_list() == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, [True, False, False, False]),
        ({"abs_tol": 1e-9}, [True, True, False, False]),
    ],
    ids=["exact", "abs_tol"],
)
def test_the_namespace_method_passes_its_options_on(
    kwargs: dict[str, Any], expected: list[bool]
) -> None:
    dtype, same, nudged, moved = _KINDS["polygon"]
    reversed_ring = [same[0][::-1]]
    df = pl.DataFrame(
        [
            _series("a", dtype, [same] * 4),
            _series("b", dtype, [same, nudged, moved, reversed_ring]),
        ]
    )

    out = df.select(pl.col("a").geo.is_equal("b", **kwargs))

    assert out["a"].to_list() == expected


@pytest.mark.parametrize("value", [-1.0, math.nan])
@pytest.mark.parametrize("name", ["abs_tol", "rel_tol"])
def test_a_tolerance_cannot_be_negative_or_nan(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"`{name}` must be non-negative and not NaN"):
        geo.is_equal("a", "b", **{name: value})


def test_different_types_are_refused() -> None:
    df = pl.DataFrame(
        [
            _series("a", PointXY(), [_LINE[0]]),
            _series("b", LineStringXY(), [_LINE]),
        ]
    )

    with pytest.raises(TypeError, match="same type"):
        df.select(geo.is_equal("a", "b"))


def test_different_dimensions_are_refused() -> None:
    xyz = pl.Series("b", [{"x": 0.0, "y": 0.0, "z": 0.0}]).ext.to(PointXYZ())
    df = pl.DataFrame([_series("a", PointXY(), [_LINE[0]]), xyz])

    with pytest.raises(TypeError, match="same type"):
        df.select(geo.is_equal("a", "b"))


def test_different_crss_are_refused() -> None:
    df = pl.DataFrame(
        [_series("a", PointXY(), [_LINE[0]]), _series("b", PointXY(), [_LINE[0]])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"), geo.set_crs("b", "EPSG:3857"))

    with pytest.raises(ValueError, match="different CRSs"):
        df.select(geo.is_equal("a", "b"))


def test_unequal_lengths_are_refused() -> None:
    a = _series("a", PointXY(), [_LINE[0], _LINE[1]])
    b = _series("b", PointXY(), [_LINE[0], _LINE[1], _LINE[2]])

    with pytest.raises((ShapeError, ComputeError)):
        pl.select(geo.is_equal(a, b))
