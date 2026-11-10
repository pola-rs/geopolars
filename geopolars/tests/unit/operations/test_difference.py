"""The difference of two geometries, forwarded to GEOS."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
import shapely
from polars.exceptions import ComputeError, ShapeError
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PointXYM,
    PointXYZ,
    PolygonXY,
    PolygonXYM,
    PolygonXYZ,
    PolygonXYZM,
)

if TYPE_CHECKING:
    from polars._typing import EngineType

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})

Coord = dict[str, float]


def _xy(*xy: tuple[float, float]) -> list[Coord]:
    return [{"x": x, "y": y} for x, y in xy]


def _series(name: str, dtype: GeoArrowType, rows: list[object]) -> pl.Series:
    """`rows` as an XY `dtype`."""
    storage: pl.DataType = _XY
    for _ in range(dtype._nesting):
        storage = pl.List(storage)
    return pl.Series(name, rows, dtype=storage).ext.to(dtype)


def _chunked(series: list[pl.Series]) -> pl.Series:
    out = pl.concat(series, rechunk=False)
    assert out.n_chunks() == len(series)
    return out


def _wkt(name: str, dtype: GeoArrowType, wkts: list[str | None]) -> pl.Series:
    out = pl.select(geo.from_wkt(pl.Series(wkts, dtype=pl.String), type(dtype)))
    return out.to_series().alias(name)


def _to_wkt(geometries: pl.Series) -> list[str | None]:
    return pl.select(geo.to_wkt(geometries)).to_series().to_list()


def _storage(geometries: pl.Series) -> list[Any]:
    return geometries.ext.storage().to_list()


def _difference(
    a: tuple[GeoArrowType, list[str | None]],
    b: tuple[GeoArrowType, list[str | None]],
    **kwargs: Any,
) -> pl.Series:
    df = pl.DataFrame([_wkt("a", *a), _wkt("b", *b)])
    return df.select(geo.difference("a", "b", **kwargs))["a"]


def _is_ccw(out: pl.Series) -> list[bool]:
    return pl.select(geo.is_ccw(out)).to_series().to_list()


def _assert_geometries(
    out: pl.Series, expected: list[str | None], *, tolerance: float = 0.0
) -> None:
    """`out` covers the same points as `expected`, row by row, where GEOS may store them otherwise.
    With a `tolerance`, a vertex may be that far off.
    """
    assert out.null_count() == expected.count(None)
    got = [None if wkt is None else shapely.from_wkt(wkt) for wkt in _to_wkt(out)]
    for row, (got_row, wkt) in enumerate(zip(got, expected, strict=True)):
        if wkt is None:
            assert got_row is None, f"row {row}"
            continue
        want = shapely.from_wkt(wkt)
        assert got_row is not None, f"row {row}"
        if want.is_empty:
            assert got_row.is_empty, f"row {row}: {got_row}"
        else:
            # Densified, so that the distance is also measured along the edges,
            # and not only from the vertices, which a different shape can share.
            # The distance only compares the rings, so it cannot tell a hole from a part.
            # The area does, and moving every vertex by `tolerance`
            # changes it by at most about `tolerance` times the perimeter.
            perimeter = shapely.length(got_row) + shapely.length(want)
            close = tolerance > 0 and (
                shapely.hausdorff_distance(got_row, want, densify=0.1) <= tolerance
                and abs(shapely.area(got_row) - shapely.area(want))
                <= tolerance * perimeter
            )
            assert shapely.equals(got_row, want) or close, (
                f"row {row}: {got_row} != {want}"
            )


# Counter-clockwise, so that a copy can be told apart from what GEOS gives (clockwise).
_SQUARE = "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"
# `_SQUARE`, running clockwise.
_CW = "POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))"
_OVERLAPPING = "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"
# What is left of `_SQUARE` once `_OVERLAPPING` is taken away.
_L = "POLYGON ((0 0, 2 0, 2 1, 1 1, 1 2, 0 2, 0 0))"
_WIDE = "POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0))"
# Cuts `_WIDE` in two.
_STRIP = "POLYGON ((1 -1, 2 -1, 2 4, 1 4, 1 -1))"
_LINE = "LINESTRING (0 0, 4 0)"
# Cuts `_LINE` in two.
_BOX_ON_LINE = "POLYGON ((1 -1, 2 -1, 2 1, 1 1, 1 -1))"
# Crosses itself at (1, 0).
_LOOP = "LINESTRING (0 0, 2 0, 2 1, 1 1, 1 -1)"
_FAR = "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))"

_KINDS = (
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
)
_ORACLE_DTYPES = {
    "Point": PointXY(),
    "MultiPoint": MultiPointXY(),
    "LineString": LineStringXY(),
    "MultiLineString": MultiLineStringXY(),
    "Polygon": PolygonXY(),
    "MultiPolygon": MultiPolygonXY(),
}


def _expected_kind(a_kind: str, splits: bool) -> str:
    if a_kind == "Point" or a_kind.startswith("Multi") or not splits:
        return a_kind
    return f"Multi{a_kind}"


@pytest.mark.parametrize("allow_splitting", [False, True])
@pytest.mark.parametrize("b_kind", _KINDS)
@pytest.mark.parametrize("a_kind", _KINDS)
def test_the_output_kind_follows_the_table(
    a_kind: str, b_kind: str, allow_splitting: bool
) -> None:
    a = pl.Series("a", [], dtype=_ORACLE_DTYPES[a_kind])
    b = pl.Series("b", [], dtype=_ORACLE_DTYPES[b_kind])
    lf = pl.LazyFrame([a, b]).select(
        geo.difference("a", "b", allow_splitting=allow_splitting)
    )

    dtype = lf.collect_schema()["a"]

    assert type(dtype).__name__ == f"{_expected_kind(a_kind, allow_splitting)}XY"


def test_overlapping_polygons_leave_what_a_does_not_share() -> None:
    out = _difference(
        (PolygonXY(), [_SQUARE, _SQUARE]), (PolygonXY(), [_OVERLAPPING, _SQUARE])
    )

    assert out.dtype == PolygonXY()
    _assert_geometries(out, [_L, "POLYGON EMPTY"])


def test_a_line_or_point_takes_nothing_away_from_a_polygon() -> None:
    a = (PolygonXY(), [_SQUARE])

    _assert_geometries(
        _difference(a, (LineStringXY(), ["LINESTRING (-1 1, 3 1)"])), [_SQUARE]
    )
    _assert_geometries(_difference(a, (PointXY(), ["POINT (1 1)"])), [_SQUARE])


def test_a_point_takes_nothing_away_from_a_line() -> None:
    out = _difference((LineStringXY(), [_LINE]), (PointXY(), ["POINT (2 0)"]))

    assert out.dtype == LineStringXY()
    _assert_geometries(out, [_LINE])


# A pair that splits on each dimension, with the number of parts, and what is left.
_SPLITS = {
    "polygons": (
        (PolygonXY(), _WIDE),
        (PolygonXY(), _STRIP),
        2,
        "MULTIPOLYGON (((0 0, 1 0, 1 3, 0 3, 0 0)), ((2 0, 3 0, 3 3, 2 3, 2 0)))",
    ),
    "lines": (
        (LineStringXY(), _LINE),
        (PolygonXY(), _BOX_ON_LINE),
        2,
        "MULTILINESTRING ((0 0, 1 0), (2 0, 4 0))",
    ),
}


@pytest.mark.parametrize("dimension", _SPLITS)
def test_a_split_raises_without_allow_splitting(dimension: str) -> None:
    a, b, parts, _ = _SPLITS[dimension]

    with pytest.raises(ComputeError, match=f"split into {parts} parts"):
        _difference((a[0], [a[1]]), (b[0], [b[1]]))


@pytest.mark.parametrize("dimension", _SPLITS)
def test_a_split_gives_null_with_ignore_per_row_errors(dimension: str) -> None:
    a, b, _, _ = _SPLITS[dimension]

    out = _difference(
        (a[0], [a[1], a[1]]),
        (b[0], [b[1], _FAR]),
        ignore_per_row_errors=True,
    )

    _assert_geometries(out, [None, a[1]])


@pytest.mark.parametrize("dimension", _SPLITS)
def test_a_split_gives_every_part_with_allow_splitting(dimension: str) -> None:
    a, b, _, left = _SPLITS[dimension]

    out = _difference((a[0], [a[1]]), (b[0], [b[1]]), allow_splitting=True)

    assert type(out.dtype).__name__.startswith("Multi")
    _assert_geometries(out, [left])


def test_a_line_only_gains_a_vertex_where_another_line_crosses_it() -> None:
    out = _difference(
        (LineStringXY(), [_LINE]), (LineStringXY(), ["LINESTRING (2 -1, 2 1)"])
    )

    assert out.dtype == LineStringXY()
    assert _to_wkt(out) == ["LINESTRING(0 0,2 0,4 0)"]


def test_a_line_crossing_itself_splits_even_where_b_does_not_touch_it() -> None:
    with pytest.raises(ComputeError, match="split into 3 parts"):
        _difference((LineStringXY(), [_LOOP]), (PointXY(), ["POINT (9 9)"]))


def test_a_point_gives_itself_only_outside() -> None:
    points = ["POINT (0.5 0.5)", "POINT (2 1)", "POINT (5 5)"]

    out = _difference((PointXY(), points), (PolygonXY(), [_SQUARE] * 3))

    assert out.dtype == PointXY()
    assert _to_wkt(out) == ["POINT EMPTY", "POINT EMPTY", "POINT(5 5)"]


def test_a_multipoint_keeps_the_points_outside() -> None:
    out = _difference(
        (
            MultiPointXY(),
            ["MULTIPOINT ((1 1), (5 5), (0 0))", None, "MULTIPOINT ((1 1))"],
        ),
        (PolygonXY(), [_SQUARE] * 3),
    )

    assert out.dtype == MultiPointXY()
    _assert_geometries(out, ["MULTIPOINT ((5 5))", None, "MULTIPOINT EMPTY"])


def test_the_exterior_runs_counter_clockwise_with_ccw_outer() -> None:
    a = (PolygonXY(), [_SQUARE])
    b = (PolygonXY(), [_OVERLAPPING])

    assert _is_ccw(_difference(a, b)) == [False]
    assert _is_ccw(_difference(a, b, ccw_outer=True)) == [True]
    a, b = (PolygonXY(), [_WIDE]), (PolygonXY(), [_STRIP])
    assert _is_ccw(_difference(a, b, allow_splitting=True)) == [False]
    split = _difference(a, b, allow_splitting=True, ccw_outer=True)
    assert split.dtype == MultiPolygonXY()
    assert _is_ccw(split) == [True]


@pytest.mark.parametrize("ccw_outer", [False, True])
def test_a_hole_is_cut(ccw_outer: bool) -> None:
    big = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"
    inner = "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"

    out = _difference((PolygonXY(), [big]), (PolygonXY(), [inner]), ccw_outer=ccw_outer)

    holed = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 3, 3 3, 3 1, 1 1))"
    _assert_geometries(out, [holed])
    assert _is_ccw(out) == [ccw_outer]


def test_an_empty_a_gives_an_empty_geometry() -> None:
    out = _difference((PolygonXY(), ["POLYGON EMPTY"]), (PolygonXY(), [_SQUARE]))
    assert _to_wkt(out) == ["POLYGON EMPTY"]

    empty_point = {"x": math.nan, "y": math.nan}
    df = pl.DataFrame(
        [_series("a", PointXY(), [empty_point]), _wkt("b", PolygonXY(), [_SQUARE])]
    )
    assert _to_wkt(df.select(geo.difference("a", "b"))["a"]) == ["POINT EMPTY"]


@pytest.mark.parametrize("ccw_outer", [False, True])
def test_an_empty_b_gives_a_as_it_is(ccw_outer: bool) -> None:
    a = _wkt("a", PolygonXY(), [_SQUARE, _CW, _SQUARE])
    b = _wkt("b", PolygonXY(), ["POLYGON EMPTY", "POLYGON EMPTY", _OVERLAPPING])

    out = pl.DataFrame([a, b]).select(geo.difference("a", "b", ccw_outer=ccw_outer))

    assert _storage(out["a"])[:2] == _storage(a)[:2]
    # Only the row that went through GEOS is turned.
    assert _is_ccw(out["a"])[2] == ccw_outer


def _coords(row: Any) -> list[Coord]:
    if row is None:
        return []
    if isinstance(row, dict):
        return [row]
    return [coord for part in row for coord in _coords(part)]


def _xy_only(row: Any) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return {"x": row["x"], "y": row["y"]}
    return [_xy_only(part) for part in row]


# Each of these would come back otherwise from GEOS:
# re-oriented, without its duplicate point, noded or split.
_AS_IS = {
    "Point": (PointXY(), ["POINT (1 2)"]),
    "MultiPoint": (MultiPointXY(), ["MULTIPOINT ((3 3), (1 1), (1 1))"]),
    "LineString": (LineStringXY(), [_LOOP]),
    "MultiLineString": (
        MultiLineStringXY(),
        ["MULTILINESTRING ((4 0, 0 0), (2 -1, 2 1))"],
    ),
    "Polygon": (
        PolygonXY(),
        [_SQUARE, "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))"],
    ),
    "MultiPolygon": (
        MultiPolygonXY(),
        ["MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((5 5, 5 6, 6 6, 6 5, 5 5)))"],
    ),
}


@pytest.mark.parametrize(
    ("b_dtype", "empty"),
    [(PolygonXY(), "POLYGON EMPTY"), (PolygonXYZ(), "POLYGON Z EMPTY")],
    ids=["b_xy", "z_only_on_b"],
)
@pytest.mark.parametrize("ccw_outer", [False, True])
@pytest.mark.parametrize("allow_splitting", [False, True])
@pytest.mark.parametrize("kind", _AS_IS)
def test_an_empty_b_gives_a_as_it_is_for_every_kind(
    kind: str,
    allow_splitting: bool,
    ccw_outer: bool,
    b_dtype: GeoArrowType,
    empty: str,
) -> None:
    dtype, wkts = _AS_IS[kind]
    a = _wkt("a", dtype, wkts)
    b = _wkt("b", b_dtype, [empty] * len(wkts))

    out = pl.DataFrame([a, b]).select(
        geo.difference("a", "b", allow_splitting=allow_splitting, ccw_outer=ccw_outer)
    )["a"]

    out_kind = _expected_kind(kind, allow_splitting)
    has_z = b_dtype == PolygonXYZ()
    assert type(out.dtype).__name__ == out_kind + ("XYZ" if has_z else "XY")
    # A copy into a multi kind is its only part.
    expected = [[row] if out_kind != kind else row for row in _storage(a)]
    got = _storage(out)
    if has_z:
        assert all(math.isnan(c["z"]) for row in got for c in _coords(row))
        got = [_xy_only(row) for row in got]
    assert got == expected


def test_an_empty_b_never_splits_a() -> None:
    out = _difference((LineStringXY(), [_LOOP]), (PolygonXY(), ["POLYGON EMPTY"]))

    assert out.dtype == LineStringXY()
    assert _to_wkt(out) == _to_wkt(_wkt("a", LineStringXY(), [_LOOP]))


def test_an_empty_point_takes_nothing_away() -> None:
    empty_point = {"x": math.nan, "y": math.nan}
    a = _wkt("a", LineStringXY(), [_LOOP])
    df = pl.DataFrame([a, _series("b", PointXY(), [empty_point])])

    out = df.select(geo.difference("a", "b"))["a"]

    assert _storage(out) == _storage(a)


@pytest.mark.parametrize(
    "b",
    [
        _series("b", PolygonXY(), [[[]]]),
        _series("b", MultiLineStringXY(), [[[]]]),
        _series("b", MultiPolygonXY(), [[[]]]),
        _series("b", MultiPolygonXY(), [[[[]]]]),
    ],
    ids=["polygon", "multilinestring", "multipolygon", "multipolygon_with_a_ring"],
)
def test_a_b_whose_parts_have_no_vertices_takes_nothing_away(b: pl.Series) -> None:
    assert pl.select(geo.is_empty(b)).to_series().to_list() == [True]
    loop = _wkt("a", LineStringXY(), [_LOOP])
    square = _wkt("a", PolygonXY(), [_SQUARE])

    for a in (loop, square):
        out = pl.DataFrame([a, b]).select(geo.difference("a", "b"))["a"]
        assert _storage(out) == _storage(a)


@pytest.mark.parametrize("ccw_outer", [False, True])
def test_a_single_empty_b_gives_every_row_as_it_is(ccw_outer: bool) -> None:
    # Both orientations, as one of them equals what GEOS gives, turned or not.
    many = _wkt("many", PolygonXY(), [_SQUARE, None, _CW, _WIDE])
    one = _wkt("one", PolygonXY(), ["POLYGON EMPTY"])

    out = pl.DataFrame(many).select(geo.difference("many", one, ccw_outer=ccw_outer))

    assert _storage(out["many"]) == _storage(many)


@pytest.mark.parametrize(
    ("a_dtype", "allow_splitting"),
    [(PolygonXY(), False), (PolygonXY(), True), (MultiPolygonXY(), False)],
    ids=["polygon", "polygon_split", "multipolygon"],
)
def test_copied_rows_line_up_with_computed_rows_in_one_chunk(
    a_dtype: GeoArrowType, allow_splitting: bool
) -> None:
    # The copies are rows 2, 4 and 6, between rows without polygons and rows from GEOS,
    # so that a copy indexed by anything but its row is turned.
    a_rows = [
        None,
        "POLYGON EMPTY",
        _SQUARE,
        _SQUARE,
        _CW,
        _SQUARE,
        _CW,
        _SQUARE,
        _SQUARE,
    ]
    b_rows = [
        _SQUARE,
        _OVERLAPPING,
        "POLYGON EMPTY",
        _OVERLAPPING,
        "POLYGON EMPTY",
        _FAR,
        "POLYGON EMPTY",
        None,
        _SQUARE,
    ]
    a = _wkt("a", a_dtype, a_rows)
    b = _wkt("b", PolygonXY(), b_rows)

    out = pl.DataFrame([a, b]).select(
        geo.difference("a", "b", allow_splitting=allow_splitting, ccw_outer=True)
    )["a"]

    wrapped = allow_splitting and a_dtype == PolygonXY()
    got, given = _storage(out), _storage(a)
    for row in (2, 4, 6):
        assert got[row] == ([given[row]] if wrapped else given[row]), f"row {row}"
    ccw = _is_ccw(out)
    assert ccw[3] is True
    assert ccw[5] is True
    assert got[0] is None
    assert got[7] is None
    assert got[1] == []
    assert got[8] == []


@pytest.mark.parametrize("ccw_outer", [False, True])
def test_a_single_a_is_copied_into_every_row_with_an_empty_b(ccw_outer: bool) -> None:
    one = _wkt("one", PolygonXY(), [_SQUARE])
    many = _chunked(
        [
            _wkt("many", PolygonXY(), ["POLYGON EMPTY", _OVERLAPPING]),
            _wkt("many", PolygonXY(), [None, "POLYGON EMPTY"]),
        ]
    )

    out = pl.DataFrame(many).select(geo.difference(one, "many", ccw_outer=ccw_outer))

    [square] = _storage(one)
    got = _storage(out.to_series())
    assert got[0] == square
    assert got[2] is None
    assert got[3] == square
    assert _is_ccw(out.to_series())[1] is ccw_outer


def test_a_sliced_chunk_copies_its_own_rows() -> None:
    def frame(a: list[str], b: list[str]) -> pl.DataFrame:
        return pl.DataFrame([_wkt("a", PolygonXY(), a), _wkt("b", PolygonXY(), b)])

    # The padding is empty where the slice is not, and the other way round.
    pad_a = ["POLYGON ((9 9, 9 10, 10 10, 10 9, 9 9))"] * 2
    pad_b = [_OVERLAPPING, "POLYGON EMPTY"]
    rows_a = [_SQUARE, _CW, _SQUARE]
    rows_b = ["POLYGON EMPTY", _OVERLAPPING, "POLYGON EMPTY"]
    whole = frame(pad_a + rows_a + pad_a, pad_b + rows_b + pad_b)
    sliced = whole.slice(len(pad_a), len(rows_a))
    expr = geo.difference("a", "b", ccw_outer=True)

    out = sliced.select(expr)

    assert_frame_equal(out, frame(rows_a, rows_b).select(expr))
    assert _storage(out["a"])[0] == _storage(sliced["a"])[0]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_the_engines_agree_on_copied_rows(engine: EngineType) -> None:
    part = pl.DataFrame(
        [
            _wkt("a", PolygonXY(), [None, _SQUARE, _CW, _SQUARE, _SQUARE]),
            _wkt(
                "b",
                PolygonXY(),
                [_SQUARE, "POLYGON EMPTY", "POLYGON EMPTY", _OVERLAPPING, None],
            ),
        ]
    )
    df = pl.concat([part] * 3, rechunk=False)
    one = _wkt("one", PolygonXY(), [_SQUARE])

    for expr in (
        geo.difference("a", "b", ccw_outer=True),
        geo.difference(one, "b", ccw_outer=True).alias("one_against_b"),
    ):
        out = df.lazy().select(expr).collect(engine=engine)
        assert_frame_equal(out, df.select(expr))


_SQUARE_M = "POLYGON M ((0 0 10, 2 0 20, 2 2 30, 0 2 40, 0 0 10))"


@pytest.mark.parametrize(
    ("a", "b", "out_dtype", "z"),
    [
        ((PolygonXYM(), _SQUARE_M), (PolygonXY(), "POLYGON EMPTY"), PolygonXY(), None),
        (
            (PolygonXYM(), _SQUARE_M),
            (PolygonXYZ(), "POLYGON Z EMPTY"),
            PolygonXYZ(),
            math.nan,
        ),
        (
            (
                PolygonXYZM(),
                "POLYGON ZM ((0 0 1 10, 2 0 1 20, 2 2 1 30, 0 2 1 40, 0 0 1 10))",
            ),
            (PolygonXYM(), "POLYGON M EMPTY"),
            PolygonXYZ(),
            1.0,
        ),
        (
            (PolygonXYZ(), "POLYGON Z ((0 0 1, 2 0 1, 2 2 1, 0 2 1, 0 0 1))"),
            (PolygonXY(), "POLYGON EMPTY"),
            PolygonXYZ(),
            1.0,
        ),
        (
            (PointXYM(), "POINT M (0 0 10)"),
            (PolygonXYZ(), "POLYGON Z EMPTY"),
            PointXYZ(),
            math.nan,
        ),
    ],
    ids=["m_on_a", "m_on_a_z_on_b", "zm_on_a_m_on_b", "z_on_a", "point_m_on_a_z_on_b"],
)
def test_an_empty_b_drops_m_and_fills_z(
    a: tuple[GeoArrowType, str],
    b: tuple[GeoArrowType, str],
    out_dtype: GeoArrowType,
    z: float | None,
) -> None:
    a_series = _wkt("a", a[0], [a[1]])

    out = pl.DataFrame([a_series, _wkt("b", b[0], [b[1]])]).select(
        geo.difference("a", "b")
    )["a"]

    assert out.dtype == out_dtype
    [row] = _storage(out)
    assert _xy_only(row) == _xy_only(_storage(a_series)[0])
    coords = _coords(row)
    assert all("m" not in c for c in coords)
    if z is None:
        return
    zs = [c["z"] for c in coords]
    if math.isnan(z):
        assert all(math.isnan(v) for v in zs)
    else:
        assert zs == [z] * len(zs)


@pytest.mark.parametrize("z_first", [True, False])
def test_z_from_either_side_is_filled_in(z_first: bool) -> None:
    flat_z = "POLYGON Z ((1 -1 7, 2 -1 7, 2 1 7, 1 1 7, 1 -1 7))"
    if z_first:
        a = (LineStringXYZ(), ["LINESTRING Z (0 0 7, 4 0 7)"])
        b = (PolygonXY(), [_BOX_ON_LINE])
    else:
        a = (LineStringXY(), [_LINE])
        b = (PolygonXYZ(), [flat_z])

    out = _difference(a, b, allow_splitting=True)

    got = shapely.from_wkt(_to_wkt(out)[0])
    assert shapely.equals(got, shapely.from_wkt(_SPLITS["lines"][3]))
    assert set(shapely.get_coordinates(got, include_z=True)[:, 2].tolist()) == {7.0}


def test_z_of_a_is_interpolated_where_b_cuts_it() -> None:
    a = (LineStringXYZ(), ["LINESTRING Z (0 0 1, 4 0 5)"])

    out = _difference(a, (PolygonXY(), [_BOX_ON_LINE]), allow_splitting=True)

    got = shapely.from_wkt(_to_wkt(out)[0])
    z = shapely.get_coordinates(got, include_z=True)[:, 2].tolist()
    assert sorted(z) == [1.0, 2.0, 3.0, 5.0]


def test_polygons_with_z_give_polygons_with_z() -> None:
    with_z = "POLYGON Z ((0 0 1, 2 0 1, 2 2 1, 0 2 1, 0 0 1))"

    out = _difference((PolygonXYZ(), [with_z]), (PolygonXY(), [_OVERLAPPING]))

    assert out.dtype == PolygonXYZ()
    got = shapely.from_wkt(_to_wkt(out)[0])
    assert shapely.equals(got, shapely.from_wkt(_L))
    assert set(shapely.get_coordinates(got, include_z=True)[:, 2].tolist()) == {1.0}


def test_m_is_dropped() -> None:
    out = _difference((PointXYM(), ["POINT M (5 5 3)"]), (PolygonXY(), [_SQUARE]))
    assert out.dtype == PointXY()
    assert _to_wkt(out) == ["POINT(5 5)"]

    # The m differs from the z on every vertex, so neither can pass for the other.
    a = (
        PolygonXYZM(),
        ["POLYGON ZM ((0 0 1 10, 2 0 1 20, 2 2 1 30, 0 2 1 40, 0 0 1 10))"],
    )
    out = _difference(a, (LineStringXYM(), ["LINESTRING M (-1 1 5, 3 1 5)"]))
    assert out.dtype == PolygonXYZ()
    got = shapely.from_wkt(_to_wkt(out)[0])
    assert shapely.equals(got, shapely.from_wkt(_SQUARE))
    assert set(shapely.get_coordinates(got, include_z=True)[:, 2].tolist()) == {1.0}


def test_a_null_geometry_on_either_side_gives_a_null_result() -> None:
    out = _difference(
        (PolygonXY(), [None, _SQUARE, None]),
        (PolygonXY(), [_SQUARE, None, None]),
    )

    assert out.to_list() == [None, None, None]


def test_against_a_single_null_every_row_is_null_of_the_kind_of_a() -> None:
    many = _wkt("many", LineStringXY(), ["LINESTRING (0 0, 1 1)", None])
    one = _wkt("one", PolygonXY(), [None])
    df = pl.DataFrame(many)

    out = df.select(geo.difference("many", one))["many"]
    assert out.to_list() == [None, None]
    assert out.dtype == LineStringXY()

    out = df.select(geo.difference(one, "many", allow_splitting=True)).to_series()
    assert out.to_list() == [None, None]
    assert out.dtype == MultiPolygonXY()


def test_columns_chunked_differently_are_paired_row_by_row() -> None:
    a = _chunked(
        [
            _wkt("a", PolygonXY(), [_SQUARE]),
            _wkt("a", PolygonXY(), [None, _SQUARE, _SQUARE]),
        ]
    )
    b = _chunked(
        [
            _wkt("b", PolygonXY(), [_OVERLAPPING, _OVERLAPPING, _SQUARE]),
            _wkt("b", PolygonXY(), [None]),
        ]
    )

    out = pl.DataFrame([a, b]).select(geo.difference("a", "b"))["a"]

    _assert_geometries(out, [_L, None, "POLYGON EMPTY", None])


def test_a_single_geometry_keeps_its_side() -> None:
    lines = _chunked(
        [
            _wkt("many", LineStringXY(), ["LINESTRING (-1 1, 3 1)", None]),
            _wkt("many", LineStringXY(), ["LINESTRING (5 5, 6 6)"]),
        ]
    )
    square = _wkt("one", PolygonXY(), [_SQUARE])
    df = pl.DataFrame(lines)

    out = df.select(geo.difference("many", square, allow_splitting=True))["many"]
    assert out.dtype == MultiLineStringXY()
    pieces = "MULTILINESTRING ((-1 1, 0 1), (2 1, 3 1))"
    _assert_geometries(out, [pieces, None, "MULTILINESTRING ((5 5, 6 6))"])
    out = df.select(geo.difference(square, "many")).to_series()
    assert out.dtype == PolygonXY()
    _assert_geometries(out, [_SQUARE, None, _SQUARE])


# GEOS fails to build a linestring of a single vertex.
_ONE_VERTEX_LINE = _xy((0, 0))
# GEOS raises on this one, but not on every geometry that is not finite.
_NOT_FINITE_LINE = _xy((0, 0), (math.nan, 1), (2, 2))
_FINITE_LINE = _xy((0, 0), (1, 0), (1, 1))
_AROUND = "POLYGON ((-1 -1, 3 -1, 3 3, -1 3, -1 -1))"


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_a_row_paired_with_a_null_never_reaches_geos(early_finite_check: bool) -> None:
    df = pl.DataFrame(
        [
            _wkt("a", PolygonXY(), [None, None, _AROUND]),
            _series(
                "b", LineStringXY(), [_ONE_VERTEX_LINE, _NOT_FINITE_LINE, _FINITE_LINE]
            ),
        ]
    )

    out = df.select(geo.difference("a", "b", early_finite_check=early_finite_check))

    _assert_geometries(out["a"], [None, None, _AROUND])


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_a_row_paired_with_an_empty_a_never_reaches_geos(
    early_finite_check: bool,
) -> None:
    df = pl.DataFrame(
        [
            # `POLYGON (EMPTY)` has a ring, but no vertices.
            _wkt(
                "a",
                PolygonXY(),
                ["POLYGON EMPTY", "POLYGON EMPTY", "POLYGON (EMPTY)", _AROUND],
            ),
            _series(
                "b",
                LineStringXY(),
                [_ONE_VERTEX_LINE, _NOT_FINITE_LINE, _ONE_VERTEX_LINE, _FINITE_LINE],
            ),
        ]
    )

    out = df.select(geo.difference("a", "b", early_finite_check=early_finite_check))

    expected = ["POLYGON EMPTY", "POLYGON EMPTY", "POLYGON EMPTY", _AROUND]
    _assert_geometries(out["a"], expected)


@pytest.mark.parametrize("early_finite_check", [False, True])
@pytest.mark.parametrize(
    ("one", "empty"),
    [
        (_wkt("one", PolygonXY(), ["POLYGON EMPTY"]), "POLYGON EMPTY"),
        (_series("one", PointXY(), [{"x": math.nan, "y": math.nan}]), "POINT EMPTY"),
    ],
    ids=["polygon", "point"],
)
def test_against_a_single_empty_a_no_row_reaches_geos(
    one: pl.Series, empty: str, early_finite_check: bool
) -> None:
    many = _series("many", LineStringXY(), [_ONE_VERTEX_LINE, _NOT_FINITE_LINE])
    df = pl.DataFrame(many)

    out = df.select(
        geo.difference(one, "many", early_finite_check=early_finite_check)
    ).to_series()

    assert _to_wkt(out) == [empty, empty]


def _not_finite_lines(**kwargs: bool) -> pl.Series:
    """A line that is not finite, then one that is, each minus a square around both."""
    df = pl.DataFrame(
        [
            _series("a", LineStringXY(), [_NOT_FINITE_LINE, _FINITE_LINE]),
            _wkt("b", PolygonXY(), [_AROUND] * 2),
        ]
    )
    return df.select(geo.difference("a", "b", **kwargs))["a"]


def test_geos_raises_on_a_line_that_is_not_finite() -> None:
    with pytest.raises(ComputeError, match="NaN/Inf"):
        _not_finite_lines()


def test_geos_failing_on_a_row_gives_null_with_ignore_per_row_errors() -> None:
    out = _not_finite_lines(ignore_per_row_errors=True)

    _assert_geometries(out, [None, "LINESTRING EMPTY"])


def test_the_early_finite_check_raises_before_geos() -> None:
    with pytest.raises(ComputeError, match="GEOS expects finite x and y"):
        _not_finite_lines(early_finite_check=True)
    out = _not_finite_lines(early_finite_check=True, ignore_per_row_errors=True)
    _assert_geometries(out, [None, "LINESTRING EMPTY"])


_OUTER = _xy((0, 0), (4, 0), (4, 4), (0, 4), (0, 0))
_NOT_FINITE_HOLE = _xy((1, 1), (1, 2), (math.nan, 2), (2, 1), (1, 1))


@pytest.mark.parametrize("early_finite_check", [False, True])
@pytest.mark.parametrize(
    "a",
    [
        _series("a", LineStringXY(), [_FINITE_LINE, _NOT_FINITE_LINE]),
        _series("a", MultiPolygonXY(), [[[_OUTER]], [[_OUTER, _NOT_FINITE_HOLE]]]),
    ],
    ids=["line", "hole_of_a_multipolygon"],
)
def test_an_empty_b_still_fails_on_an_a_that_is_not_finite(
    a: pl.Series, early_finite_check: bool
) -> None:
    df = pl.DataFrame([a, _wkt("b", PolygonXY(), ["POLYGON EMPTY"] * 2)])

    def run(**kwargs: bool) -> pl.Series:
        expr = geo.difference("a", "b", early_finite_check=early_finite_check, **kwargs)
        return df.select(expr)["a"]

    with pytest.raises(ComputeError, match="GEOS expects finite x and y"):
        run()
    assert _storage(run(ignore_per_row_errors=True)) == [_storage(a)[0], None]


@pytest.mark.parametrize("single_first", [True, False])
def test_ignore_per_row_errors_with_a_single_geometry(single_first: bool) -> None:
    def run(one: pl.Series, many: pl.Series, **kwargs: bool) -> pl.Series:
        a, b = (one, "many") if single_first else ("many", one)
        return pl.DataFrame(many).select(geo.difference(a, b, **kwargs)).to_series()

    around = _wkt("one", PolygonXY(), [_AROUND])
    lines = _series("many", LineStringXY(), [_NOT_FINITE_LINE, _FINITE_LINE])
    out = run(around, lines, ignore_per_row_errors=True)
    _assert_geometries(out, [None, _AROUND if single_first else "LINESTRING EMPTY"])

    bad = _series("one", LineStringXY(), [_NOT_FINITE_LINE])
    squares = _wkt("many", PolygonXY(), [_SQUARE, _SQUARE])
    with pytest.raises(ComputeError, match="NaN/Inf"):
        run(bad, squares)
    assert run(bad, squares, ignore_per_row_errors=True).to_list() == [None, None]


def test_the_namespace_method_passes_its_options_on() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_WIDE]), _wkt("b", PolygonXY(), [_STRIP])]
    )

    out = df.select(pl.col("a").geo.difference("b", ignore_per_row_errors=True))
    assert out["a"].to_list() == [None]
    out = df.select(
        pl.col("a").geo.difference(
            "b", allow_splitting=True, ccw_outer=True, early_finite_check=True
        )
    )
    assert out["a"].dtype == MultiPolygonXY()
    assert _is_ccw(out["a"]) == [True]


def test_the_crs_of_a_is_kept() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (1 1)"])]
    ).with_columns(geo.set_crs("a", "EPSG:3857"), geo.set_crs("b", "EPSG:3857"))

    out = df.select(geo.difference("a", "b"))

    assert out["a"].dtype._crs() == df["a"].dtype._crs()


def test_different_crss_are_refused() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"), geo.set_crs("b", "EPSG:3857"))

    with pytest.raises(ValueError, match="different CRSs"):
        df.select(geo.difference("a", "b"))


def test_a_crs_on_only_one_side_is_refused() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"))

    with pytest.raises(ValueError, match="only one of the geometries"):
        df.select(geo.difference("a", "b"))


def test_unequal_lengths_are_refused() -> None:
    a = _wkt("a", PointXY(), ["POINT (0 0)", "POINT (1 1)"])
    b = _wkt("b", PolygonXY(), [_SQUARE, _SQUARE, _SQUARE])

    with pytest.raises((ShapeError, ComputeError)):
        pl.select(geo.difference(a, b))


_DIMENSION = {"Point": 0, "LineString": 1, "Polygon": 2}


def _random_shape(rng: random.Random, kind: str) -> Any:
    """A small shape on a grid, so that shapes often meet, touch and overlap.
    Polygons include a U-shape, so that their differences can split.
    The parts of a multipolygon are far apart, so that it stays valid.
    """
    u = "POLYGON ((0 0, 3 0, 3 3, 2 3, 2 1, 1 1, 1 3, 0 3, 0 0))"

    def vertex() -> tuple[int, int]:
        return rng.randint(0, 4), rng.randint(0, 4)

    def line() -> Any:
        return shapely.LineString([vertex() for _ in range(rng.randint(2, 4))])

    def polygon() -> Any:
        if rng.random() < 0.3:
            return shapely.affinity.translate(
                shapely.from_wkt(u), rng.randint(0, 2), rng.randint(0, 2)
            )
        hull = shapely.convex_hull(shapely.MultiPoint([vertex() for _ in range(4)]))
        return hull if isinstance(hull, shapely.Polygon) else polygon()

    match kind:
        case "Point":
            return shapely.Point(vertex())
        case "MultiPoint":
            return shapely.MultiPoint([vertex() for _ in range(rng.randint(1, 4))])
        case "LineString":
            return line()
        case "MultiLineString":
            return shapely.MultiLineString([line() for _ in range(rng.randint(1, 3))])
        case "Polygon":
            return polygon()
        case "MultiPolygon":
            parts = [polygon(), shapely.affinity.translate(polygon(), 10, 0)]
            return shapely.MultiPolygon(parts[: rng.randint(1, 2)])
    raise AssertionError(kind)


def _kept_parts(difference: Any, dimension: int) -> list[Any]:
    parts = shapely.get_parts(difference)
    return [
        p for p in parts if not p.is_empty and shapely.get_dimensions(p) == dimension
    ]


@pytest.mark.parametrize("b_kind", _ORACLE_DTYPES)
@pytest.mark.parametrize("a_kind", _ORACLE_DTYPES)
def test_matches_shapely(a_kind: str, b_kind: str) -> None:
    # The random shapes are never empty, so every row goes through GEOS, as in shapely.
    dimension = _DIMENSION[a_kind.removeprefix("Multi")]
    rng = random.Random(f"{a_kind} {b_kind}")
    a = [_random_shape(rng, a_kind) for _ in range(200)]
    b = [_random_shape(rng, b_kind) for _ in range(200)]
    df = pl.DataFrame(
        [
            _wkt("a", _ORACLE_DTYPES[a_kind], list(shapely.to_wkt(a))),
            _wkt("b", _ORACLE_DTYPES[b_kind], list(shapely.to_wkt(b))),
        ]
    )
    kept = [_kept_parts(shapely.difference(x, y), dimension) for x, y in zip(a, b)]

    def differences(**kwargs: bool) -> pl.Series:
        out = df.select(geo.difference("a", "b", ignore_per_row_errors=True, **kwargs))
        return out["a"]

    every_part = differences(allow_splitting=True)
    assert any(len(parts) > 0 for parts in kept), "the cases should not all be empty"
    expected = [shapely.GeometryCollection(parts) for parts in kept]
    # Shapely may bring another GEOS, which can compute a crossing a bit differently,
    # and merge other parts. A split is therefore checked against our own parts.
    _assert_geometries(
        every_part,
        list(shapely.to_wkt(expected, rounding_precision=-1)),
        tolerance=1e-9,
    )
    # Parts that are apart cannot be merged by any GEOS, so their number has to agree.
    for row, (wkt, parts) in enumerate(zip(_to_wkt(every_part), kept, strict=True)):
        apart = not any(
            p.intersects(q) for i, p in enumerate(parts) for q in parts[i + 1 :]
        )
        if wkt is not None and apart:
            got = shapely.from_wkt(wkt)
            count = 0 if got.is_empty else shapely.get_num_geometries(got)
            assert count == len(parts), f"row {row}"
    if _expected_kind(a_kind, splits=False).startswith("Multi") or a_kind == "Point":
        return
    # A lower dimension takes nothing away (see `test_a_line_or_point_takes_nothing_away_from_a_polygon`).
    if dimension <= _DIMENSION[b_kind.removeprefix("Multi")]:
        assert any(len(parts) > 1 for parts in kept), "some rows should split"
    expected = [
        None if shapely.get_num_geometries(shapely.from_wkt(wkt)) > 1 else wkt
        for wkt in _to_wkt(every_part)
    ]
    _assert_geometries(differences(), expected)
