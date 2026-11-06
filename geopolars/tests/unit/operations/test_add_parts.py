"""Adding parts to geometries from another column."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal

from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    LineStringXYZ,
    LineStringXYZM,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PolygonXY,
)
from tests.unit.conftest import Dimension

if TYPE_CHECKING:
    from polars._typing import EngineType

RD = "EPSG:28992"

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})

Vertex = dict[str, float]
Line = list[Vertex]
Polygon = list[Line]


def _vertex(x: float, y: float) -> Vertex:
    return {"x": x, "y": y}


def _line(*xy: tuple[float, float]) -> Line:
    return [_vertex(x, y) for x, y in xy]


_A = _line((0, 0), (1, 1))
_B = _line((2, 2), (3, 3), (4, 2))
_C = _line((5, 5), (6, 6))
_EMPTY_POINT = _vertex(math.nan, math.nan)
_P = [_line((0, 0), (1, 0), (0, 1), (0, 0))]
_Q = [_line((5, 5), (6, 5), (5, 6), (5, 5)), _line((5.1, 5.1), (5.2, 5.1), (5.1, 5.2))]


def _column(
    name: str, rows: list[object], storage: pl.DataType, dtype: pl.DataType
) -> pl.Series:
    return pl.Series(name, rows, dtype=storage).ext.to(dtype)


def _listed(
    name: str, rows: list[object], storage: pl.DataType, dtype: pl.DataType
) -> pl.Series:
    series = pl.Series(name, rows, dtype=pl.List(storage))
    return series.list.eval(pl.element().ext.to(dtype))


def _points(
    *rows: Vertex | None, name: str = "geometry", crs: str | None = None
) -> pl.Series:
    return _column(name, list(rows), _XY, PointXY(crs=crs))


def _multi_points(*rows: Line | None, name: str = "geometry") -> pl.Series:
    return _column(name, list(rows), pl.List(_XY), MultiPointXY())


def _listed_points(*rows: list[Vertex | None] | None) -> pl.Series:
    return _listed("parts", list(rows), _XY, PointXY())


def _lines(
    *rows: Line | None, name: str = "geometry", crs: str | None = None
) -> pl.Series:
    return _column(name, list(rows), pl.List(_XY), LineStringXY(crs=crs))


def _multi_lines(*rows: list[Line] | None, name: str = "geometry") -> pl.Series:
    return _column(name, list(rows), pl.List(pl.List(_XY)), MultiLineStringXY())


def _listed_lines(*rows: list[Line | None] | None, crs: str | None = None) -> pl.Series:
    return _listed("parts", list(rows), pl.List(_XY), LineStringXY(crs=crs))


def _polygons(*rows: Polygon | None, name: str = "geometry") -> pl.Series:
    return _column(name, list(rows), pl.List(pl.List(_XY)), PolygonXY())


def _multi_polygons(*rows: list[Polygon] | None, name: str = "geometry") -> pl.Series:
    return _column(name, list(rows), pl.List(pl.List(pl.List(_XY))), MultiPolygonXY())


def _listed_polygons(*rows: list[Polygon | None] | None) -> pl.Series:
    return _listed("parts", list(rows), pl.List(pl.List(_XY)), PolygonXY())


def _bare(rows: list[object], lists: int) -> pl.Series:
    """`rows` as bare coordinates, nested `lists` lists deep."""
    storage: pl.DataType = _XY
    for _ in range(lists):
        storage = pl.List(storage)
    return pl.Series("parts", rows, dtype=storage)


def _nested(
    rows: list[object], storage: pl.DataType, dtype: pl.DataType, lists: int
) -> pl.Series:
    """`rows` as `dtype`s nested `lists` lists deep (at least one)."""
    for _ in range(lists):
        storage = pl.List(storage)
    expr = pl.element().ext.to(dtype)
    for _ in range(lists - 1):
        expr = pl.element().list.eval(expr)
    return pl.Series("parts", rows, dtype=storage).list.eval(expr)


def _of_points(rows: list[object], lists: int, crs: str | None = None) -> pl.Series:
    return _nested(rows, _XY, PointXY(crs=crs), lists)


def _added(geometry: pl.Series, parts: pl.Series, **kwargs: object) -> pl.Series:
    df = pl.DataFrame([geometry, parts.alias("parts")])
    return df.select(geo.add_parts("geometry", "parts", **kwargs)).to_series()


def _assert_storage_equal_with_nan(out: pl.Series, expected: list[object]) -> None:
    storage = out.ext.storage()
    assert_series_equal(storage, pl.Series(storage.name, expected, storage.dtype))


@pytest.mark.parametrize(
    ("geometry", "parts", "dtype", "expected"),
    [
        (
            _lines(_A, _B),
            _lines(_C, _A, name="parts"),
            MultiLineStringXY(),
            [[_A, _C], [_B, _A]],
        ),
        (
            _lines(_A, _B),
            _listed_lines([_B, _C], []),
            MultiLineStringXY(),
            [[_A, _B, _C], [_B]],
        ),
        (
            _lines(_A, _B),
            _multi_lines([_B, _C], [], name="parts"),
            MultiLineStringXY(),
            [[_A, _B, _C], [_B]],
        ),
        (
            _multi_lines([_A, _B], []),
            _lines(_C, _A, name="parts"),
            MultiLineStringXY(),
            [[_A, _B, _C], [_A]],
        ),
        (
            _multi_lines([_A, _B], []),
            _listed_lines([_C], [_A, _B]),
            MultiLineStringXY(),
            [[_A, _B, _C], [_A, _B]],
        ),
        (
            _multi_lines([_A, _B], [_C]),
            _multi_lines([_C], [_A], name="parts"),
            MultiLineStringXY(),
            [[_A, _B, _C], [_C, _A]],
        ),
        (_points(_A[0]), _points(_A[1], name="parts"), MultiPointXY(), [_A]),
        (
            _points(_A[0]),
            _listed_points([_B[0], _B[1]]),
            MultiPointXY(),
            [[_A[0], *_B[:2]]],
        ),
        (
            _points(_A[0]),
            _multi_points(_B, name="parts"),
            MultiPointXY(),
            [[_A[0], *_B]],
        ),
        (
            _multi_points(_A),
            _points(_C[0], name="parts"),
            MultiPointXY(),
            [[*_A, _C[0]]],
        ),
        (_multi_points(_A), _listed_points([_C[1]]), MultiPointXY(), [[*_A, _C[1]]]),
        (
            _multi_points(_A),
            _multi_points(_B, name="parts"),
            MultiPointXY(),
            [_A + _B],
        ),
        (
            _polygons(_P),
            _polygons(_Q, name="parts"),
            MultiPolygonXY(),
            [[_P, _Q]],
        ),
        (
            _polygons(_P),
            _listed_polygons([_Q, _P]),
            MultiPolygonXY(),
            [[_P, _Q, _P]],
        ),
        (
            _polygons(_P),
            _multi_polygons([_Q], name="parts"),
            MultiPolygonXY(),
            [[_P, _Q]],
        ),
        (
            _multi_polygons([_P, _Q]),
            _polygons(_P, name="parts"),
            MultiPolygonXY(),
            [[_P, _Q, _P]],
        ),
        (
            _multi_polygons([_Q]),
            _listed_polygons([_P, _Q]),
            MultiPolygonXY(),
            [[_Q, _P, _Q]],
        ),
        (
            _multi_polygons([_Q]),
            _multi_polygons([_P], name="parts"),
            MultiPolygonXY(),
            [[_Q, _P]],
        ),
    ],
    ids=[
        "linestring: a linestring",
        "linestring: linestrings in a list",
        "linestring: a multilinestring",
        "multilinestring: a linestring",
        "multilinestring: linestrings in a list",
        "multilinestring: a multilinestring",
        "point: a point",
        "point: points in a list",
        "point: a multipoint",
        "multipoint: a point",
        "multipoint: points in a list",
        "multipoint: a multipoint",
        "polygon: a polygon",
        "polygon: polygons in a list",
        "polygon: a multipolygon",
        "multipolygon: a polygon",
        "multipolygon: polygons in a list",
        "multipolygon: a multipolygon",
    ],
)
def test_parts_are_added_after_those_the_geometry_has(
    geometry: pl.Series, parts: pl.Series, dtype: pl.DataType, expected: list[object]
) -> None:
    out = _added(geometry, parts)

    assert out.dtype == dtype
    assert out.ext.storage().to_list() == expected


@pytest.mark.parametrize(
    "parts",
    [_listed_lines([_B], crs=RD), _bare([_B], 1)],
    ids=["parts in the same crs", "parts as bare coordinates"],
)
def test_the_crs_of_the_geometry_is_kept(parts: pl.Series) -> None:
    out = _added(_lines(_A, crs=RD), parts)

    assert out.dtype == MultiLineStringXY(crs=RD)
    assert out.ext.storage().to_list() == [[_A, _B]]


@pytest.mark.parametrize(
    ("geometry", "parts", "expected"),
    [
        (_points(_A[0]), _bare([_B[0]], 0), [[_A[0], _B[0]]]),
        (_points(_A[0]), _bare([_B[:2]], 1), [[_A[0], *_B[:2]]]),
        (_lines(_A), _of_points([_B], 1), [[_A, _B]]),
        (_lines(_A), _of_points([[_B, _C]], 2), [[_A, _B, _C]]),
        (_lines(_A), _bare([_B], 1), [[_A, _B]]),
        (_lines(_A), _bare([[_B, _C]], 2), [[_A, _B, _C]]),
        (_lines(_A), _multi_points(_B, name="parts"), [[_A, _B]]),
        (
            _lines(_A),
            _nested([[_B, _C]], pl.List(_XY), MultiPointXY(), 1),
            [[_A, _B, _C]],
        ),
        (_polygons(_P), _of_points([_Q], 2), [[_P, _Q]]),
        (_polygons(_P), _of_points([[_Q, _P]], 3), [[_P, _Q, _P]]),
        (_polygons(_P), _listed_lines(_Q), [[_P, _Q]]),
        (_polygons(_P), _nested([[_Q]], pl.List(_XY), LineStringXY(), 2), [[_P, _Q]]),
        (_polygons(_P), _bare([_Q], 2), [[_P, _Q]]),
        (_polygons(_P), _bare([[_Q, _P]], 3), [[_P, _Q, _P]]),
        (_polygons(_P), _multi_lines(_Q, name="parts"), [[_P, _Q]]),
        (_polygons(_P), _nested([_Q], pl.List(_XY), MultiPointXY(), 1), [[_P, _Q]]),
        (
            _polygons(_P),
            _nested([[_Q, _P]], pl.List(_XY), MultiPointXY(), 2),
            [[_P, _Q, _P]],
        ),
        (
            _polygons(_P),
            _nested([[_Q, _P]], pl.List(pl.List(_XY)), MultiLineStringXY(), 1),
            [[_P, _Q, _P]],
        ),
    ],
    ids=[
        "point: bare coordinates",
        "point: bare coordinates in a list",
        "linestring: points in a list",
        "linestring: points in 2 lists",
        "linestring: bare coordinates in a list",
        "linestring: bare coordinates in 2 lists",
        "linestring: a multipoint",
        "linestring: multipoints in a list",
        "polygon: points in 2 lists",
        "polygon: points in 3 lists",
        "polygon: linestrings in a list",
        "polygon: linestrings in 2 lists",
        "polygon: bare coordinates in 2 lists",
        "polygon: bare coordinates in 3 lists",
        "polygon: a multilinestring",
        "polygon: multipoints in a list",
        "polygon: multipoints in 2 lists",
        "polygon: multilinestrings in a list",
    ],
)
def test_a_part_can_be_given_as_its_points_or_rings(
    geometry: pl.Series, parts: pl.Series, expected: list[object]
) -> None:
    out = _added(geometry, parts)

    assert out.ext.storage().to_list() == expected


def test_no_rows() -> None:
    out = _added(_lines(), _listed_lines())

    assert out.dtype == MultiLineStringXY()
    assert out.to_list() == []


def test_parts_are_not_checked() -> None:
    open_ring = [_line((0, 0), (1, 0), (0, 1))]

    out = _added(_polygons(open_ring), _listed_polygons([[_A], []]))

    assert out.ext.storage().to_list() == [[open_ring, [_A], []]]


def test_adding_the_parts_back_rebuilds_the_multilinestrings(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    lines = (
        ring_coords.group_by("polygon", "ring", maintain_order=True)
        .agg(dimension.point())
        .select("polygon", "ring", dimension.linestring())
        .group_by("polygon", maintain_order=True)
        .agg(
            pl.col("line").filter(pl.col("ring") == 0).first().alias("first"),
            pl.col("line").filter(pl.col("ring") > 0).alias("rest"),
        )
    )

    out = lines.select(geo.add_parts("first", "rest").alias("multilinestring"))

    assert_frame_equal(out, dimension.multilinestrings(ring_coords))


def test_adding_the_parts_back_rebuilds_the_multipoints(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    points = (
        line_coords.select("line", dimension.point())
        .group_by("line", maintain_order=True)
        .agg(
            pl.col("point").first().alias("first"),
            pl.col("point").slice(1).alias("rest"),
        )
    )

    out = points.select(geo.add_parts("first", "rest").alias("multipoint"))

    assert_frame_equal(out, dimension.multipoints(line_coords))


def test_adding_the_parts_back_rebuilds_the_multipolygons(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    polygons = (
        multipolygon_coords.group_by(
            "multipolygon", "polygon", "ring", maintain_order=True
        )
        .agg(dimension.point())
        .select("multipolygon", "polygon", dimension.linestring())
        .group_by("multipolygon", "polygon", maintain_order=True)
        .agg("line")
        .select("multipolygon", dimension.polygon())
        .group_by("multipolygon", maintain_order=True)
        .agg(
            pl.col("polygon").first().alias("first"),
            pl.col("polygon").slice(1).alias("rest"),
        )
    )

    out = polygons.select(geo.add_parts("first", "rest").alias("multipolygon"))

    assert_frame_equal(out, dimension.multipolygons(multipolygon_coords))


@pytest.mark.parametrize(
    ("geometry", "parts", "dropped", "nulled"),
    [
        (_lines(_A), _listed_lines(None), [[_A]], [None]),
        (_lines(_A), _listed_lines([_B, None, _C]), [[_A, _B, _C]], [None]),  # type: ignore[list-item]
        (_lines(_A), _listed_lines([None]), [[_A]], [None]),  # type: ignore[list-item]
        (_lines(_A), _listed_lines([_B, [None], _C]), [[_A, _B, _C]], [None]),  # type: ignore[list-item]
        (
            _lines(_A),
            _listed_lines([[{"x": None, "y": 1.0}], _B]),  # type: ignore[list-item]
            [[_A, _B]],
            [None],
        ),
        (_lines(_A), _lines(None, name="parts"), [[_A]], [None]),
        (_lines(_A), _lines([_B[0], None], name="parts"), [[_A]], [None]),  # type: ignore[list-item]
        (_lines(_A), _listed_lines([]), [[_A]], [[_A]]),
        (_points(_A[0]), _listed_points(None), [[_A[0]]], [None]),
        (_points(_A[0]), _listed_points([None, _B[0]]), [[_A[0], _B[0]]], [None]),
        (
            _points(_A[0]),
            _listed_points([{"x": 1.0, "y": None}, _B[0]]),  # type: ignore[list-item]
            [[_A[0], _B[0]]],
            [None],
        ),
        (_polygons(_P), _listed_polygons([None, _Q]), [[_P, _Q]], [None]),
        (_polygons(_P), _listed_polygons([_Q, [_P[0], None]]), [[_P, _Q]], [None]),  # type: ignore[list-item]
        (_polygons(_P), _listed_polygons([[[None]], _Q]), [[_P, _Q]], [None]),  # type: ignore[list-item]
        (_lines(None), _listed_lines([_B]), [None], [None]),
        (_points(None), _listed_points([_B[0]]), [None], [None]),
    ],
    ids=[
        "a null row of linestrings",
        "a null linestring",
        "only a null linestring",
        "a linestring with a null vertex",
        "a linestring with a null axis",
        "a single null linestring",
        "a single linestring with a null vertex",
        "an empty list is no null",
        "a null row of points",
        "a null point",
        "a point with a null axis",
        "a null polygon",
        "a polygon with a null ring",
        "a polygon with a null vertex",
        "a null linestring geometry",
        "a null point geometry",
    ],
)
def test_a_null_part_is_dropped_or_nulls_the_result(
    geometry: pl.Series,
    parts: pl.Series,
    dropped: list[object],
    nulled: list[object],
) -> None:
    out_dropped = _added(geometry, parts, null_parts="drop")
    out_nulled = _added(geometry, parts, null_parts="null")

    assert out_dropped.ext.storage().to_list() == dropped
    assert out_nulled.ext.storage().to_list() == nulled


def test_null_parts_null_the_result_by_default() -> None:
    out = _added(_lines(_A, _A), _listed_lines([_B, None], [_B]))  # type: ignore[list-item]

    assert out.ext.storage().to_list() == [None, [_A, _B]]


@pytest.mark.parametrize(
    ("geometry", "dtype", "part"),
    [
        (_points(_A[0], None), MultiPointXY(), _A[0]),
        (_lines(_A, None), MultiLineStringXY(), _A),
        (_polygons(_P, None), MultiPolygonXY(), _P),
    ],
    ids=["point", "linestring", "polygon"],
)
def test_an_untyped_null_adds_nothing_or_nulls_the_result(
    geometry: pl.Series, dtype: pl.DataType, part: object
) -> None:
    df = pl.DataFrame([geometry])

    dropped = df.select(
        geo.add_parts("geometry", pl.lit(None), null_parts="drop")
    ).to_series()
    nulled = df.select(
        geo.add_parts("geometry", pl.lit(None), null_parts="null")
    ).to_series()

    assert dropped.dtype == dtype
    assert dropped.ext.storage().to_list() == [[part], None]
    assert nulled.ext.storage().to_list() == [None, None]


@pytest.mark.parametrize(
    ("geometry", "parts", "kept", "dropped"),
    [
        (
            _multi_lines([[]]),
            _listed_lines([[], _B, []]),
            [[[], [], _B, []]],
            # The geometry's own empty part stays.
            [[[], _B]],
        ),
        (
            _lines(_A, _A),
            _lines([], _B, name="parts"),
            [[_A, []], [_A, _B]],
            [[_A], [_A, _B]],
        ),
        (
            _points(_A[0]),
            _listed_points([_EMPTY_POINT, _B[0]]),
            [[_A[0], _EMPTY_POINT, _B[0]]],
            [[_A[0], _B[0]]],
        ),
        (
            _points(_A[0]),
            _listed_points([_vertex(math.nan, 1.0), _vertex(1.0, math.nan)]),
            [[_A[0], _vertex(math.nan, 1.0), _vertex(1.0, math.nan)]],
            [[_A[0], _vertex(math.nan, 1.0), _vertex(1.0, math.nan)]],
        ),
        (
            _polygons(_P),
            _listed_polygons([[], _Q, [[]]]),
            [[_P, [], _Q, [[]]]],
            [[_P, _Q]],
        ),
        (
            _points(_EMPTY_POINT),
            _listed_points([_B[0]]),
            [[_EMPTY_POINT, _B[0]]],
            [[_B[0]]],
        ),
        (
            _points(_A[0], _EMPTY_POINT, _C[0]),
            _listed_points([_B[0]], [_B[1]], [_B[2]]),
            [[_A[0], _B[0]], [_EMPTY_POINT, _B[1]], [_C[0], _B[2]]],
            [[_A[0], _B[0]], [_B[1]], [_C[0], _B[2]]],
        ),
        (_points(_EMPTY_POINT), _listed_points([]), [[_EMPTY_POINT]], [[]]),
        (_lines([]), _listed_lines([_B]), [[[], _B]], [[_B]]),
        (_polygons([]), _listed_polygons([_Q]), [[[], _Q]], [[_Q]]),
        (_polygons([[]]), _listed_polygons([_Q]), [[[[]], _Q]], [[_Q]]),
    ],
    ids=[
        "linestrings in a list",
        "a single linestring",
        "points in a list",
        "a point with only some axes nan is not empty",
        "polygons in a list",
        "an empty point geometry",
        "an empty point geometry between others",
        "an empty point geometry without parts",
        "an empty linestring geometry",
        "an empty polygon geometry",
        "a polygon geometry with an empty ring",
    ],
)
def test_empty_parts_are_kept_or_dropped(
    geometry: pl.Series,
    parts: pl.Series,
    kept: list[object],
    dropped: list[object],
) -> None:
    _assert_storage_equal_with_nan(_added(geometry, parts), kept)
    _assert_storage_equal_with_nan(_added(geometry, parts, empty_parts="drop"), dropped)


@pytest.mark.parametrize(
    ("geometry", "parts", "expected"),
    [
        (
            _points(_A[0]),
            _listed_points([_B[0], None, _C[0], _EMPTY_POINT, _A[1]]),
            [_A[0], _B[0], _C[0], _A[1]],
        ),
        (
            _lines(_A),
            _listed_lines([None, _B, [], _C, None]),  # type: ignore[list-item]
            [_A, _B, _C],
        ),
        (_polygons(_P), _listed_polygons([_Q, None, _P, [], _Q]), [_P, _Q, _P, _Q]),
    ],
    ids=["points", "linestrings", "polygons"],
)
def test_null_and_empty_parts_are_left_out_and_the_rest_stays_in_order(
    geometry: pl.Series, parts: pl.Series, expected: list[object]
) -> None:
    out = _added(geometry, parts, null_parts="drop", empty_parts="drop")

    assert out.ext.storage().to_list() == [expected]


def test_empty_parts_are_dropped_while_null_parts_null_the_result() -> None:
    out = _added(
        _lines(_A, _A),
        _listed_lines([[], _B], [None, _B]),
        null_parts="null",
        empty_parts="drop",
    )

    assert out.ext.storage().to_list() == [[_A, _B], None]


def test_the_parts_left_out_keep_z_and_m_of_the_others_in_place() -> None:
    def line(*values: float) -> list[dict[str, float]]:
        return [{"x": v, "y": v, "z": v * 10, "m": v * 100} for v in values]

    xyzm = pl.Struct({axis: pl.Float64 for axis in "xyzm"})
    geometry = _column("geometry", [line(0, 1)], pl.List(xyzm), LineStringXYZM())
    parts = _listed(
        "parts",
        [[line(2, 3), None, [], line(4, 5), None, line(6, 7)]],
        pl.List(xyzm),
        LineStringXYZM(),
    )

    out = _added(geometry, parts, null_parts="drop", empty_parts="drop")

    assert out.ext.storage().to_list() == [
        [line(0, 1), line(2, 3), line(4, 5), line(6, 7)]
    ]


@pytest.mark.parametrize(
    ("parts", "kwargs"),
    [
        (_lines(_C), {}),
        (
            _listed_lines([None, _C, []]),  # type: ignore[list-item]
            {"null_parts": "drop", "empty_parts": "drop"},
        ),
    ],
    ids=["a part", "a list with null and empty parts"],
)
def test_a_single_row_of_parts_is_added_to_every_row(
    parts: pl.Series, kwargs: dict[str, str]
) -> None:
    df = pl.DataFrame([_lines(_A, _B)])

    out = df.select(geo.add_parts("geometry", parts, **kwargs)).to_series()  # type: ignore[arg-type]

    assert out.ext.storage().to_list() == [[_A, _C], [_B, _C]]


def test_a_single_geometry_gets_every_row_of_parts() -> None:
    df = pl.DataFrame([_lines(_B, _C, name="parts")])

    out = df.select(geo.add_parts(_lines(_A), "parts")).to_series()

    assert out.ext.storage().to_list() == [[_A, _B], [_A, _C]]


def test_rows_of_different_lengths_are_refused() -> None:
    df = pl.DataFrame([_lines(_A, _B)])

    with pytest.raises(ComputeError, match="2 rows but the parts have 3"):
        df.select(geo.add_parts("geometry", _lines(_A, _B, _C, name="parts")))


@pytest.mark.parametrize(
    ("geometry", "parts", "expected"),
    [
        (
            _multi_lines([_C], [_A], None, [_B]),
            _listed_lines([_A], [_B, _C], [_A], [None, _A]),  # type: ignore[list-item]
            [[_A, _B, _C], None, [_B, _A]],
        ),
        (
            _multi_points([_C[0]], [_A[0]], None, [_B[0]]),
            _listed_points([_A[0]], [_B[0], None], [_A[0]], [None, _A[0]]),
            [[_A[0], _B[0]], None, [_B[0], _A[0]]],
        ),
        (
            _multi_polygons([_Q], [_P], None, [_P]),
            _listed_polygons([_P], [_Q, None], [_P], [[[None]], _Q]),  # type: ignore[list-item]
            [[_P, _Q], None, [_P, _Q]],
        ),
    ],
    ids=["linestrings", "points", "polygons"],
)
def test_sliced_columns(
    geometry: pl.Series, parts: pl.Series, expected: list[object]
) -> None:
    df = pl.DataFrame([geometry, parts]).slice(1, 3)

    out = df.select(geo.add_parts("geometry", "parts", null_parts="drop"))

    assert out.to_series().ext.storage().to_list() == expected


@pytest.mark.parametrize(
    ("geometry", "parts", "kwargs", "expected"),
    [
        # The empty part before the slice shifts every part after it by one,
        # until the parts are trimmed to the slice.
        (
            _lines(_A, _A),
            _listed_lines([[]], [_B, [], _C]),
            {"empty_parts": "drop"},
            [[_A, _B, _C]],
        ),
        (
            _points(_A[0], _A[0]),
            _listed_points([None, _EMPTY_POINT], [_B[0], None, _EMPTY_POINT, _C[0]]),
            {"null_parts": "drop", "empty_parts": "drop"},
            [[_A[0], _B[0], _C[0]]],
        ),
        (
            _polygons(_P, _P),
            _listed_polygons([None, []], [_Q, [[None]], [[]], _P]),  # type: ignore[list-item]
            {"null_parts": "drop", "empty_parts": "drop"},
            [[_P, _Q, _P]],
        ),
    ],
    ids=["empty linestrings", "null and empty points", "null and empty polygons"],
)
def test_parts_left_out_before_a_slice_stay_left_out(
    geometry: pl.Series,
    parts: pl.Series,
    kwargs: dict[str, str],
    expected: list[object],
) -> None:
    df = pl.DataFrame([geometry, parts]).slice(1, 1)

    out = df.select(geo.add_parts("geometry", "parts", **kwargs))  # type: ignore[arg-type]

    assert out.to_series().ext.storage().to_list() == expected


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_every_chunk_gets_its_parts(engine: EngineType) -> None:
    df = pl.concat(
        [
            pl.DataFrame([_lines(_A), _listed_lines([_B, None])]),  # type: ignore[list-item]
            pl.DataFrame([_lines(None, _B), _listed_lines([_C], [_C])]),
        ],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    out = (
        df.lazy()
        .select(geo.add_parts("geometry", "parts", null_parts="drop"))
        .collect(engine=engine)
    )

    assert out.to_series().ext.storage().to_list() == [[_A, _B], None, [_B, _C]]


def test_the_namespace_forwards() -> None:
    df = pl.DataFrame([_lines(_A), _listed_lines([_B]).alias("parts")])

    assert_frame_equal(
        df.select(pl.col("geometry").geo.add_parts("parts", empty_parts="drop")),
        df.select(geo.add_parts("geometry", "parts", empty_parts="drop")),
    )


def test_a_geometry_column_without_a_geometry_is_refused() -> None:
    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        _added(_points(_A[0]).ext.storage(), _points(_B[0], name="parts"))


_XYZ_VERTICES = pl.Series(
    "parts",
    [[{"x": 1.0, "y": 1.0, "z": 0.0}]],
    dtype=pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})),
)


@pytest.mark.parametrize(
    "parts",
    [_XYZ_VERTICES.ext.to(LineStringXYZ()), _XYZ_VERTICES],
    ids=["linestring", "bare"],
)
def test_parts_of_another_dimension_are_refused(parts: pl.Series) -> None:
    lf = pl.LazyFrame([_lines(_A), parts]).select(geo.add_parts("geometry", "parts"))

    with pytest.raises(ComputeError, match="the geometry is xy but the parts are xyz"):
        lf.collect_schema()


@pytest.mark.parametrize(
    ("geometry", "parts", "match"),
    [
        (_points(_A[0], crs=RD), _points(_B[0], crs="EPSG:4326"), "different CRSs"),
        (_points(_A[0], crs=RD), _points(_B[0]), "only one"),
        (_points(_A[0]), _points(_B[0], crs=RD), "only one"),
        (_lines(_A), _listed_lines([_B], crs=RD), "only one"),
        (_lines(_A, crs=RD), _of_points([_B], 1, crs="EPSG:4326"), "different CRSs"),
    ],
    ids=[
        "different crs",
        "a crs on the geometry only",
        "a crs on the parts only",
        "a crs on the listed parts only",
        "a different crs on the points of the parts",
    ],
)
def test_parts_in_another_crs_are_refused(
    geometry: pl.Series, parts: pl.Series, match: str
) -> None:
    lf = pl.LazyFrame([geometry, parts.alias("parts")]).select(
        geo.add_parts("geometry", "parts")
    )

    with pytest.raises(ComputeError, match=match):
        lf.collect_schema()


@pytest.mark.parametrize(
    ("geometry", "parts"),
    [
        (_lines(_A), _polygons(_P, name="parts")),
        (_points(_A[0]), _lines(_A, name="parts")),
        # A single linestring is one ring, not a polygon.
        (_polygons(_P), _lines(_A, name="parts")),
        (_lines(_A), _of_points([[[_A]]], 3)),
        (_lines(_A), _bare([[[_A]]], 3)),
        (
            _polygons(_P),
            _nested([[[_P]]], pl.List(pl.List(pl.List(_XY))), MultiPolygonXY(), 1),
        ),
    ],
    ids=[
        "linestring: a polygon",
        "point: a linestring",
        "polygon: a linestring",
        "linestring: points in 3 lists",
        "linestring: bare coordinates in 3 lists",
        "polygon: multipolygons in a list",
    ],
)
def test_parts_of_another_kind_are_refused(
    geometry: pl.Series, parts: pl.Series
) -> None:
    lf = pl.LazyFrame([geometry, parts]).select(geo.add_parts("geometry", "parts"))

    with pytest.raises(ComputeError, match="expected the parts as"):
        lf.collect_schema()


@pytest.mark.parametrize(
    ("geometry", "parts", "spelling"),
    [
        (
            _points(_A[0]),
            _of_points([[_A]], 2),
            "a point can also be its bare coordinates",
        ),
        (
            _lines(_A),
            pl.Series("parts", [1.0]),
            "a linestring can also be a list of its points or a `multipoint`",
        ),
        (
            _polygons(_P),
            _of_points([_A], 1),
            "a polygon can also be a list of its rings or a `multilinestring`",
        ),
    ],
    ids=["point", "linestring", "polygon"],
)
def test_a_refusal_spells_out_the_parts_of_the_geometry(
    geometry: pl.Series, parts: pl.Series, spelling: str
) -> None:
    lf = pl.LazyFrame([geometry, parts]).select(geo.add_parts("geometry", "parts"))

    with pytest.raises(ComputeError, match=spelling):
        lf.collect_schema()


@pytest.mark.parametrize("kwargs", [{"null_parts": "skip"}, {"empty_parts": "null"}])
def test_unknown_modes_are_refused(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        geo.add_parts("geometry", "parts", **kwargs)  # type: ignore[arg-type]
