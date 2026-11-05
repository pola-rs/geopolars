"""Whether two geometries are equal: identical coordinate by coordinate, or topologically."""

from __future__ import annotations

import math
import random
from typing import Any

import polars as pl
import pytest
import shapely
from polars.exceptions import ComputeError, ShapeError

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYM,
    MultiLineStringXY,
    MultiLineStringXYZM,
    MultiPointXY,
    MultiPolygonXY,
    MultiPolygonXYZ,
    PointXY,
    PointXYM,
    PointXYZ,
    PointXYZM,
    PolygonXY,
    PolygonXYZ,
    PolygonXYZM,
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
    """`is_equal` in identical mode, with `rel_tol=1e-9` unless given."""
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
        ({"mode": "topological"}, [True, False, False, True]),
    ],
    ids=["exact", "abs_tol", "topological"],
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


def _rejects_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="`mode` to be one of"):
        geo.is_equal("a", "b", mode=mode)  # type: ignore[arg-type]


def test_an_unknown_mode_is_refused() -> None:
    _rejects_mode("exact")


@pytest.mark.parametrize(
    "option", [{"ignore_per_row_errors": True}, {"early_finite_check": True}]
)
def test_identical_refuses_the_geos_options(option: dict[str, bool]) -> None:
    with pytest.raises(ValueError, match='only applies to mode="topological"'):
        geo.is_equal("a", "b", **option)


# Topological equality


def _wkt(name: str, dtype: GeoArrowType, wkts: list[str | None]) -> pl.Series:
    out = pl.select(geo.from_wkt(pl.Series(wkts, dtype=pl.String), type(dtype)))
    return out.to_series().alias(name)


def _topologically_equal(
    a: tuple[GeoArrowType, list[str | None]],
    b: tuple[GeoArrowType, list[str | None]],
    **kwargs: bool,
) -> list[bool | None]:
    df = pl.DataFrame([_wkt("a", *a), _wkt("b", *b)])
    out = df.select(geo.is_equal("a", "b", mode="topological", **kwargs))
    return out["a"].to_list()


_SQUARE = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"


@pytest.mark.parametrize(
    "other",
    [
        "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))",
        "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
        "POLYGON ((0 0, 0.5 0, 1 0, 1 1, 0 1, 0 0))",
    ],
    ids=["another start", "reversed", "extra vertex along an edge"],
)
def test_the_same_polygon_stored_differently_is_equal(other: str) -> None:
    assert _topologically_equal((PolygonXY(), [_SQUARE]), (PolygonXY(), [other])) == [
        True
    ]


def test_a_different_polygon_is_not_equal() -> None:
    moved = "POLYGON ((0 0, 2 0, 2 1, 0 1, 0 0))"

    assert _topologically_equal((PolygonXY(), [_SQUARE]), (PolygonXY(), [moved])) == [
        False
    ]


def test_parts_in_another_order_are_equal() -> None:
    a = "MULTILINESTRING ((0 0, 1 0), (5 5, 6 6))"
    b = "MULTILINESTRING ((5 5, 6 6), (0 0, 1 0))"

    assert _topologically_equal(
        (MultiLineStringXY(), [a]), (MultiLineStringXY(), [b])
    ) == [True]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((PolygonXY(), _SQUARE), (MultiPolygonXY(), f"MULTIPOLYGON ({_SQUARE[7:]})")),
        (
            (LineStringXY(), "LINESTRING (0 0, 1 0, 2 0)"),
            (MultiLineStringXY(), "MULTILINESTRING ((0 0, 1 0), (1 0, 2 0))"),
        ),
        ((PointXY(), "POINT (1 2)"), (MultiPointXY(), "MULTIPOINT ((1 2), (1 2))")),
    ],
    ids=["polygon", "linestring split in two", "point twice"],
)
def test_a_geometry_equals_its_multi_variant(
    a: tuple[GeoArrowType, str], b: tuple[GeoArrowType, str]
) -> None:
    assert _topologically_equal((a[0], [a[1]]), (b[0], [b[1]])) == [True]
    assert _topologically_equal((b[0], [b[1]]), (a[0], [a[1]])) == [True]


def test_z_is_not_compared() -> None:
    a = "POLYGON Z ((0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 0))"
    b = "MULTIPOLYGON Z (((0 0 5, 1 0 5, 1 1 5, 0 1 5, 0 0 5)))"

    assert _topologically_equal((PolygonXYZ(), [a]), (MultiPolygonXYZ(), [b])) == [True]
    assert _topologically_equal((PolygonXY(), [_SQUARE]), (MultiPolygonXYZ(), [b])) == [
        True
    ]


def test_any_two_empty_geometries_are_equal() -> None:
    empty_point = {"x": math.nan, "y": math.nan}
    df = pl.DataFrame(
        [
            _series("a", PointXY(), [empty_point, empty_point]),
            _wkt("b", PolygonXY(), ["POLYGON EMPTY", _SQUARE]),
        ]
    )

    out = df.select(geo.is_equal("a", "b", mode="topological"))

    assert out["a"].to_list() == [True, False]


def test_an_empty_geometry_is_not_equal_to_one_with_coordinates() -> None:
    a = (LineStringXY(), ["LINESTRING EMPTY", "LINESTRING (0 0, 1 1)"])
    b = (MultiLineStringXY(), ["MULTILINESTRING ((0 0, 1 1))", "MULTILINESTRING EMPTY"])

    assert _topologically_equal(a, b) == [False, False]


def test_a_null_geometry_on_either_side_gives_a_null_result() -> None:
    a = (PolygonXY(), [_SQUARE, None, None])
    b = (MultiPolygonXY(), [None, f"MULTIPOLYGON ({_SQUARE[7:]})", None])

    assert _topologically_equal(a, b) == [None, None, None]


def test_a_single_geometry_is_compared_topologically_against_every_row() -> None:
    many = _wkt(
        "many",
        PolygonXY(),
        [_SQUARE, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))", "POLYGON EMPTY", None],
    )
    one = _wkt("one", MultiPolygonXY(), [f"MULTIPOLYGON ({_SQUARE[7:]})"])
    df = pl.DataFrame(many)
    expected = [True, True, False, None]

    out = df.select(geo.is_equal("many", one, mode="topological"))
    assert out["many"].to_list() == expected
    out = df.select(geo.is_equal(one, "many", mode="topological"))
    assert out.to_series().to_list() == expected


def test_against_a_single_null_geometry_everything_is_topologically_null() -> None:
    many = _wkt("many", PolygonXY(), [_SQUARE, "POLYGON EMPTY"])
    one = _wkt("one", PointXY(), [None])

    out = pl.DataFrame(many).select(geo.is_equal("many", one, mode="topological"))

    assert out["many"].to_list() == [None, None]


def test_columns_chunked_differently_are_compared_row_by_row() -> None:
    a = _chunked([_wkt("a", PolygonXY(), [_SQUARE]), _wkt("a", PolygonXY(), [None])])
    b = _wkt("b", MultiPolygonXY(), [f"MULTIPOLYGON ({_SQUARE[7:]})", None])

    out = pl.DataFrame([a, b]).select(geo.is_equal("a", "b", mode="topological"))

    assert out["a"].to_list() == [True, None]


def test_a_chunked_column_is_compared_topologically_against_a_single_geometry() -> None:
    reversed_square = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
    moved = "POLYGON ((0 0, 2 0, 2 1, 0 1, 0 0))"
    many = _chunked(
        [
            _wkt("many", PolygonXY(), [_SQUARE, moved]),
            _wkt("many", PolygonXY(), [None, reversed_square, "POLYGON EMPTY"]),
        ]
    )
    one = _wkt("one", MultiPolygonXY(), [f"MULTIPOLYGON ({_SQUARE[7:]})"])
    df = pl.DataFrame(many)
    expected = [True, False, None, True, False]

    out = df.select(geo.is_equal("many", one, mode="topological"))
    assert out["many"].to_list() == expected
    out = df.select(geo.is_equal(one, "many", mode="topological"))
    assert out.to_series().to_list() == expected


def test_m_is_not_compared_topologically() -> None:
    a = (PolygonXYZM(), ["POLYGON ZM ((0 0 1 1, 1 0 1 2, 1 1 1 3, 0 1 1 4, 0 0 1 1))"])
    b = (MultiPolygonXY(), [f"MULTIPOLYGON ({_SQUARE[7:]})"])

    assert _topologically_equal(a, b) == [True]
    line = (LineStringXYM(), ["LINESTRING M (0 0 1, 1 1 2)"])
    reversed_line = (MultiLineStringXYZM(), ["MULTILINESTRING ZM ((1 1 0 9, 0 0 0 9))"])
    assert _topologically_equal(line, reversed_line) == [True]


# GEOS raises on this one, but not on every geometry that is not finite.
_NOT_FINITE_LINE = _xy((0, 0), (math.nan, 1), (2, 2))


def _not_finite_lines(**kwargs: bool) -> list[bool | None]:
    """A line that is not finite, then one that is, each compared with itself."""
    rows = [_NOT_FINITE_LINE, _LINE]
    df = pl.DataFrame(
        [_series("a", LineStringXY(), rows), _series("b", LineStringXY(), rows)]
    )
    return df.select(geo.is_equal("a", "b", mode="topological", **kwargs))[
        "a"
    ].to_list()


def test_geos_raises_on_a_line_that_is_not_finite() -> None:
    with pytest.raises(ComputeError, match="NaN/Inf"):
        _not_finite_lines()


def test_geos_failing_on_a_row_gives_null_with_ignore_per_row_errors() -> None:
    assert _not_finite_lines(ignore_per_row_errors=True) == [None, True]


def test_the_early_finite_check_raises_before_geos() -> None:
    with pytest.raises(ComputeError, match="GEOS expects finite x and y"):
        _not_finite_lines(early_finite_check=True)
    assert _not_finite_lines(early_finite_check=True, ignore_per_row_errors=True) == [
        None,
        True,
    ]


def test_a_point_that_is_not_finite_is_compared_as_it_is() -> None:
    a = [
        {"x": math.nan, "y": 1.0},
        {"x": math.inf, "y": 1.0},
        {"x": math.nan, "y": 1.0},
    ]
    b = [{"x": math.nan, "y": 1.0}, {"x": math.inf, "y": 1.0}, {"x": 0.0, "y": 1.0}]
    df = pl.DataFrame([_series("a", PointXY(), a), _series("b", PointXY(), b)])

    out = df.select(geo.is_equal("a", "b", mode="topological"))

    assert out["a"].to_list() == [True, True, False]


@pytest.mark.parametrize("option", ["abs_tol", "rel_tol"])
def test_topological_refuses_a_tolerance(option: str) -> None:
    with pytest.raises(ValueError, match="does not support a tolerance"):
        geo.is_equal("a", "b", mode="topological", **{option: 0.1})


@pytest.mark.parametrize(
    ("option", "reason"),
    [
        ({"nans_equal": False}, "no option for how NaNs compare"),
        ({"unique_z": False}, "compares x and y only"),
        ({"unique_m": True}, "compares x and y only"),
    ],
)
def test_topological_refuses_changed_axis_options(
    option: dict[str, bool], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        geo.is_equal("a", "b", mode="topological", **option)


def test_topological_takes_the_axis_options_at_their_defaults() -> None:
    defaults = {"nans_equal": True, "unique_z": True, "unique_m": False}

    out = _topologically_equal(
        (PolygonXY(), [_SQUARE]), (PolygonXY(), [_SQUARE]), **defaults
    )
    assert out == [True]


def test_topological_refuses_different_crss() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"), geo.set_crs("b", "EPSG:3857"))

    with pytest.raises(ValueError, match="different CRSs"):
        df.select(geo.is_equal("a", "b", mode="topological"))


def test_topological_refuses_unequal_lengths() -> None:
    a = _wkt("a", PointXY(), ["POINT (0 0)", "POINT (1 1)"])
    b = _wkt("b", PolygonXY(), [_SQUARE, _SQUARE, _SQUARE])

    with pytest.raises((ShapeError, ComputeError)):
        pl.select(geo.is_equal(a, b, mode="topological"))


# Every kind with a few ways to store each shape, so that equal pairs are common across kinds.
_ORACLE_DTYPES = {
    "Point": PointXY(),
    "MultiPoint": MultiPointXY(),
    "LineString": LineStringXY(),
    "MultiLineString": MultiLineStringXY(),
    "Polygon": PolygonXY(),
    "MultiPolygon": MultiPolygonXY(),
}


def _shapes_of_every_kind(rng: random.Random) -> dict[str, list[Any]]:
    """For one random set of vertices, every kind of shape through them,
    each stored in a few equivalent ways.
    Every kind has an empty one too, as only those are equal across a point, a line, and a polygon.
    """
    vertices = [(rng.randint(0, 2), rng.randint(0, 2)) for _ in range(3)]
    line = shapely.LineString(vertices)
    split = shapely.MultiLineString([vertices[:2], vertices[1:]])
    hull = shapely.convex_hull(shapely.MultiPoint(vertices))
    polygons = [hull, shapely.reverse(hull), shapely.segmentize(hull, 0.5)]
    polygons = [p for p in polygons if isinstance(p, shapely.Polygon)]
    point = shapely.Point(vertices[0])
    return {
        "Point": [point, shapely.Point()],
        "MultiPoint": [
            shapely.MultiPoint([vertices[0]]),
            shapely.MultiPoint([vertices[0], vertices[0]]),
            shapely.MultiPoint(vertices),
            shapely.MultiPoint(),
        ],
        "LineString": [line, shapely.reverse(line), shapely.LineString()],
        "MultiLineString": [
            shapely.MultiLineString([line]),
            split,
            shapely.MultiLineString(),
        ],
        "Polygon": [*polygons, shapely.Polygon()],
        "MultiPolygon": [
            *(shapely.MultiPolygon([p]) for p in polygons),
            shapely.MultiPolygon(),
        ],
    }


@pytest.mark.parametrize("b_kind", _ORACLE_DTYPES)
@pytest.mark.parametrize("a_kind", _ORACLE_DTYPES)
def test_topological_equality_matches_shapely(a_kind: str, b_kind: str) -> None:
    rng = random.Random(f"{a_kind} {b_kind}")
    a: list[Any] = []
    b: list[Any] = []
    while len(a) < 200:
        shapes = _shapes_of_every_kind(rng)
        other = shapes if rng.random() < 0.5 else _shapes_of_every_kind(rng)
        if shapes[a_kind] and other[b_kind]:
            a.append(rng.choice(shapes[a_kind]))
            b.append(rng.choice(other[b_kind]))
    df = pl.DataFrame(
        [
            _wkt("a", _ORACLE_DTYPES[a_kind], list(shapely.to_wkt(a))),
            _wkt("b", _ORACLE_DTYPES[b_kind], list(shapely.to_wkt(b))),
        ]
    )

    out = df.select(geo.is_equal("a", "b", mode="topological"))["a"].to_list()

    expected = list(shapely.equals(a, b))
    assert out == expected
    assert any(expected), "the cases should not all be unequal"
