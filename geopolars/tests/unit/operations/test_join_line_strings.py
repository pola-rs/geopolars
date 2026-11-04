"""Joining linestrings end to start into one."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.exceptions import ComputeError

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    MultiLineStringXY,
    MultiLineStringXYZM,
    PointXY,
)

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_XYZM = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "m": pl.Float64})
RD = "EPSG:28992"

Coord = dict[str, float]
Line = list[Coord]


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


def _multilines(*multis: list[Line] | None, crs: str | None = None) -> pl.DataFrame:
    storage = pl.Series("g", list(multis), dtype=pl.List(pl.List(_XY)))
    return pl.DataFrame(storage.ext.to(MultiLineStringXY(crs=crs)))


def _storage(df: pl.DataFrame) -> list:
    return df.select(pl.col("g").ext.storage())["g"].to_list()


def _join(*multis: list[Line] | None, **kwargs: object) -> list:
    return _storage(_multilines(*multis).select(geo.join_line_strings("g", **kwargs)))


_AB = _line((0, 0), (1, 0))
_BC = _line((1, 0), (1, 1))
_DE = _line((5, 5), (6, 6))


def test_coinciding_parts_are_joined_keeping_both_join_vertices() -> None:
    assert _join([_AB, _BC]) == [_line((0, 0), (1, 0), (1, 0), (1, 1))]


def test_drop_repeated_joins_keeps_one_join_vertex() -> None:
    assert _join([_AB, _BC], drop_repeated_joins=True) == [
        _line((0, 0), (1, 0), (1, 1))
    ]


def test_drop_repeated_joins_keeps_the_end_of_the_earlier_part() -> None:
    next_part = _line((1 + 1e-12, 0), (1, 1))

    assert _join([_AB, next_part], abs_tol=1e-9, drop_repeated_joins=True) == [
        _line((0, 0), (1, 0), (1, 1))
    ]


def test_a_gap_nulls_the_row_by_default() -> None:
    assert _join([_AB, _DE], [_AB, _BC]) == [
        None,
        _line((0, 0), (1, 0), (1, 0), (1, 1)),
    ]


def test_a_bridged_gap_is_joined_anyway() -> None:
    assert _join([_AB, _DE], gaps="bridge") == [_line((0, 0), (1, 0), (5, 5), (6, 6))]


def test_a_split_gap_starts_a_new_part_and_coinciding_parts_merge() -> None:
    out = _multilines([_AB, _BC, _DE]).select(geo.join_line_strings("g", gaps="split"))

    assert out.schema["g"] == MultiLineStringXY()
    assert _storage(out) == [[_line((0, 0), (1, 0), (1, 0), (1, 1)), _DE]]


def test_split_drops_repeated_joins_but_not_at_a_gap() -> None:
    assert _join([_AB, _BC, _DE], gaps="split", drop_repeated_joins=True) == [
        [_line((0, 0), (1, 0), (1, 1)), _DE]
    ]


def test_bridge_drops_repeated_joins_but_not_at_a_gap() -> None:
    assert _join([_AB, _BC, _DE], gaps="bridge", drop_repeated_joins=True) == [
        _line((0, 0), (1, 0), (1, 1), (5, 5), (6, 6))
    ]


@pytest.mark.parametrize(
    ("tolerance", "joined"),
    [({}, False), ({"rel_tol": 1e-9}, True), ({"abs_tol": 1e-9}, True)],
    ids=["exact", "rel_tol", "abs_tol"],
)
def test_ends_that_are_almost_the_same_need_a_tolerance(
    tolerance: dict[str, float], joined: bool
) -> None:
    next_part = _line((1 + 1e-12, 0), (1, 1))
    expected = [*_AB, *next_part] if joined else None

    assert _join([_AB, next_part], **tolerance) == [expected]


@pytest.mark.parametrize("value", [-1.0, math.nan])
@pytest.mark.parametrize("name", ["abs_tol", "rel_tol"])
def test_a_tolerance_cannot_be_negative_or_nan(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"`{name}` must be non-negative and not NaN"):
        geo.join_line_strings("g", **{name: value})


def _xyzm(x: float, z: float, m: float) -> Coord:
    return {"x": x, "y": 0.0, "z": z, "m": m}


@pytest.mark.parametrize(
    ("flags", "joined"),
    [
        ({}, False),
        ({"unique_z": False}, True),
        ({"unique_z": False, "unique_m": True}, False),
    ],
    ids=["z compared", "z ignored", "m compared"],
)
def test_z_and_m_are_compared_as_asked(flags: dict[str, bool], joined: bool) -> None:
    first = [_xyzm(0, 0, 0), _xyzm(1, 0, 0)]
    second = [_xyzm(1, 1, 1), _xyzm(2, 1, 1)]
    storage = pl.Series("g", [[first, second]], dtype=pl.List(pl.List(_XYZM)))
    df = pl.DataFrame(storage.ext.to(MultiLineStringXYZM()))

    out = _storage(df.select(geo.join_line_strings("g", **flags)))

    assert out == [[*first, *second] if joined else None]


def test_empty_parts_are_skipped() -> None:
    assert _join([_AB, [], _BC], [[], _DE, []]) == [
        _line((0, 0), (1, 0), (1, 0), (1, 1)),
        _DE,
    ]


@pytest.mark.parametrize(
    ("gaps", "joined"),
    [
        ("null", LineStringXY()),
        ("bridge", LineStringXY()),
        ("split", MultiLineStringXY()),
    ],
)
def test_nothing_to_join_gives_an_empty_result(gaps: str, joined: GeoArrowType) -> None:
    out = _multilines([], [[], []], None).select(geo.join_line_strings("g", gaps=gaps))

    assert out.schema["g"] == joined
    assert _storage(out) == [[], [], None]


def test_split_skips_empty_parts() -> None:
    assert _join([[], _AB, _BC, []], [_AB, [], _DE], gaps="split") == [
        [_line((0, 0), (1, 0), (1, 0), (1, 1))],
        [_AB, _DE],
    ]


@pytest.mark.parametrize(
    ("gaps", "joined"),
    [
        ("null", LineStringXY(crs=RD)),
        ("bridge", LineStringXY(crs=RD)),
        ("split", MultiLineStringXY(crs=RD)),
    ],
)
def test_a_list_of_line_strings_joins_as_a_multilinestring_does(
    gaps: str, joined: GeoArrowType
) -> None:
    df = _multilines([_AB, _BC], [_AB, _DE], None, crs=RD)
    listed = df.select(geo.parts("g"))

    out = listed.select(geo.join_line_strings("g", gaps=gaps))

    assert out.schema["g"] == joined
    assert _storage(out) == _storage(df.select(geo.join_line_strings("g", gaps=gaps)))


def test_split_keeps_the_dimension() -> None:
    vertex = {"x": 0.0, "y": 0.0, "z": 1.0, "m": 2.0}
    storage = pl.Series("g", [[[vertex, vertex]]], dtype=pl.List(pl.List(_XYZM)))
    df = pl.DataFrame(storage.ext.to(MultiLineStringXYZM()))

    out = df.select(geo.join_line_strings("g", gaps="split"))

    assert out.schema["g"] == MultiLineStringXYZM()


def test_a_null_line_string_in_the_list_nulls_the_row() -> None:
    listed = pl.Series(
        "g", [[_AB, None], [_AB, _BC]], dtype=pl.List(pl.List(_XY))
    ).list.eval(pl.element().ext.to(LineStringXY()))

    out = pl.DataFrame(listed).select(geo.join_line_strings("g"))

    assert _storage(out) == [None, _line((0, 0), (1, 0), (1, 0), (1, 1))]


_ABC = _line((0, 0), (1, 0), (1, 0), (1, 1))
_ABC_DROPPED = _line((0, 0), (1, 0), (1, 1))


@pytest.mark.parametrize(
    ("gaps", "drop_repeated_joins", "last"),
    [
        ("null", False, None),
        ("null", True, None),
        ("bridge", False, [*_ABC, *_DE]),
        ("bridge", True, [*_ABC_DROPPED, *_DE]),
        ("split", False, [_ABC, _DE]),
        ("split", True, [_ABC_DROPPED, _DE]),
    ],
)
def test_null_rows_stay_null_and_leave_the_next_row_alone(
    gaps: str, drop_repeated_joins: bool, last: object
) -> None:
    listed = pl.Series(
        "g", [[_AB, _DE, None], None, [_AB, _BC, _DE]], dtype=pl.List(pl.List(_XY))
    ).list.eval(pl.element().ext.to(LineStringXY()))

    out = pl.DataFrame(listed).select(
        geo.join_line_strings("g", gaps=gaps, drop_repeated_joins=drop_repeated_joins)
    )

    assert _storage(out) == [None, None, last]


def test_a_sliced_input_is_joined_by_its_own_rows() -> None:
    df = _multilines([_DE], [_AB, _BC], [_AB, _DE]).slice(1, 2)

    assert _storage(df.select(geo.join_line_strings("g", gaps="split"))) == [
        [_line((0, 0), (1, 0), (1, 0), (1, 1))],
        [_AB, _DE],
    ]


def test_several_chunks_are_joined() -> None:
    first, second = _multilines([_AB, _BC]), _multilines([_AB, _DE])
    df = pl.concat([first, second], rechunk=False)

    out = df.select(geo.join_line_strings("g", gaps="bridge"))

    assert _storage(out) == [
        _line((0, 0), (1, 0), (1, 0), (1, 1)),
        _line((0, 0), (1, 0), (5, 5), (6, 6)),
    ]


def test_the_namespace_method_forwards() -> None:
    df = _multilines([_AB, _BC])

    assert _storage(df.select(pl.col("g").geo.join_line_strings())) == _storage(
        df.select(geo.join_line_strings("g"))
    )


@pytest.mark.parametrize("dtype", [PointXY(), LineStringXY(), pl.List(PointXY())])
def test_other_dtypes_are_refused(dtype: pl.DataType) -> None:
    df = pl.DataFrame({"g": pl.Series([], dtype=dtype)})

    with pytest.raises(ComputeError, match="expects a list of"):
        df.lazy().select(geo.join_line_strings("g")).collect_schema()


def test_an_unknown_gaps_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="`gaps` must be one of"):
        geo.join_line_strings("g", gaps="ignore")  # type: ignore[arg-type]
