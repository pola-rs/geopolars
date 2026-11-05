"""Adding parts to geometries from another column."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    LineStringXYZ,
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


def test_a_linestring_and_its_parts_become_a_multilinestring() -> None:
    out = _added(_lines(_A, crs=RD), _listed_lines([_B, _C], crs=RD))

    assert out.dtype == MultiLineStringXY(crs=RD)
    assert out.ext.storage().to_list() == [[_A, _B, _C]]


@pytest.mark.parametrize(
    ("geometry", "parts", "expected"),
    [
        (_lines(_A, _B), _lines(_C, _A, name="parts"), [[_A, _C], [_B, _A]]),
        (_lines(_A, _B), _listed_lines([_B, _C], []), [[_A, _B, _C], [_B]]),
        (
            _lines(_A, _B),
            _multi_lines([_B, _C], [], name="parts"),
            [[_A, _B, _C], [_B]],
        ),
        (
            _multi_lines([_A, _B], []),
            _lines(_C, _A, name="parts"),
            [[_A, _B, _C], [_A]],
        ),
        (
            _multi_lines([_A, _B], []),
            _listed_lines([_C], [_A, _B]),
            [[_A, _B, _C], [_A, _B]],
        ),
        (
            _multi_lines([_A, _B], [_C]),
            _multi_lines([_C], [_A], name="parts"),
            [[_A, _B, _C], [_C, _A]],
        ),
    ],
)
def test_lines(geometry: pl.Series, parts: pl.Series, expected: list[object]) -> None:
    out = _added(geometry, parts)

    assert out.dtype == MultiLineStringXY()
    assert out.ext.storage().to_list() == expected


@pytest.mark.parametrize(
    ("geometry", "parts", "expected"),
    [
        (_points(_A[0]), _points(_A[1], name="parts"), [_A]),
        (_points(_A[0]), _listed_points([_B[0], _B[1]]), [[_A[0], *_B[:2]]]),
        (_points(_A[0]), _multi_points(_B, name="parts"), [[_A[0], *_B]]),
        (_multi_points(_A), _points(_C[0], name="parts"), [[*_A, _C[0]]]),
        (_multi_points(_A), _listed_points([_C[1]]), [[*_A, _C[1]]]),
        (_multi_points(_A), _multi_points(_B, name="parts"), [_A + _B]),
    ],
)
def test_points(geometry: pl.Series, parts: pl.Series, expected: list[object]) -> None:
    out = _added(geometry, parts)

    assert out.dtype == MultiPointXY()
    assert out.ext.storage().to_list() == expected


@pytest.mark.parametrize(
    ("geometry", "parts", "expected"),
    [
        (_polygons(_P), _polygons(_Q, name="parts"), [[_P, _Q]]),
        (_polygons(_P), _listed_polygons([_Q, _P]), [[_P, _Q, _P]]),
        (_polygons(_P), _multi_polygons([_Q], name="parts"), [[_P, _Q]]),
        (_multi_polygons([_P, _Q]), _polygons(_P, name="parts"), [[_P, _Q, _P]]),
        (_multi_polygons([_Q]), _listed_polygons([_P, _Q]), [[_Q, _P, _Q]]),
        (_multi_polygons([_Q]), _multi_polygons([_P], name="parts"), [[_Q, _P]]),
    ],
)
def test_polygons(
    geometry: pl.Series, parts: pl.Series, expected: list[object]
) -> None:
    out = _added(geometry, parts)

    assert out.dtype == MultiPolygonXY()
    assert out.ext.storage().to_list() == expected


@pytest.mark.parametrize(
    ("geometry", "parts", "expected"),
    [
        (_points(_A[0]), _bare([_B[0]], 0), [[_A[0], _B[0]]]),
        (_points(_A[0]), _bare([_B[:2]], 1), [[_A[0], *_B[:2]]]),
        (_lines(_A), _of_points([_B], 1), [[_A, _B]]),
        (_lines(_A), _of_points([[_B, _C]], 2), [[_A, _B, _C]]),
        (_lines(_A), _bare([_B], 1), [[_A, _B]]),
        (_lines(_A), _bare([[_B, _C]], 2), [[_A, _B, _C]]),
        (_polygons(_P), _of_points([_Q], 2), [[_P, _Q]]),
        (_polygons(_P), _of_points([[_Q, _P]], 3), [[_P, _Q, _P]]),
        (_polygons(_P), _listed_lines(_Q), [[_P, _Q]]),
        (_polygons(_P), _nested([[_Q]], pl.List(_XY), LineStringXY(), 2), [[_P, _Q]]),
        (_polygons(_P), _bare([_Q], 2), [[_P, _Q]]),
        (_polygons(_P), _bare([[_Q, _P]], 3), [[_P, _Q, _P]]),
    ],
)
def test_parts_as_their_points_or_rings(
    geometry: pl.Series, parts: pl.Series, expected: list[object]
) -> None:
    out = _added(geometry, parts)

    assert out.ext.storage().to_list() == expected


def test_bare_coordinates_take_the_crs_of_the_geometry() -> None:
    out = _added(_lines(_A, crs=RD), _bare([_B], 1))

    assert out.dtype == MultiLineStringXY(crs=RD)


def test_a_null_point_in_a_part_of_points_is_ignored_or_propagates() -> None:
    geometry, parts = _lines(_A, _A), _of_points([_B, [_B[0], None]], 1)

    ignored = _added(geometry, parts)
    propagated = _added(geometry, parts, null_behavior="propagate")

    assert ignored.ext.storage().to_list() == [[_A, _B], [_A]]
    assert propagated.ext.storage().to_list() == [[_A, _B], None]


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


def test_null_parts_are_ignored() -> None:
    out = _added(
        _lines(None, _A, _A, _A, _A, _A),
        _listed_lines([_B], None, [], [_B, None, _C], [None], [_B, [None], _C]),  # type: ignore[list-item]
    )

    assert out.ext.storage().to_list() == [
        None,
        [_A],
        [_A],
        [_A, _B, _C],
        [_A],
        [_A, _B, _C],
    ]


def test_a_null_single_part_is_ignored() -> None:
    parts = pl.Series("parts", [None, [_vertex(1, 1), None]], dtype=pl.List(_XY))

    out = _added(_lines(_A, _A), parts.ext.to(LineStringXY()))

    assert out.ext.storage().to_list() == [[_A], [_A]]


def test_null_parts_propagate() -> None:
    out = _added(
        _lines(None, _A, _A, _A, _A),
        _listed_lines([_B], None, [], [_B, None], [_B, [None]]),  # type: ignore[list-item]
        null_behavior="propagate",
    )

    assert out.ext.storage().to_list() == [None, None, [_A], None, None]


def test_null_points_are_ignored_or_propagate() -> None:
    geometry, parts = _points(_A[0], _A[0]), _listed_points([None, _B[0]], None)

    ignored = _added(geometry, parts)
    propagated = _added(geometry, parts, null_behavior="propagate")

    assert ignored.ext.storage().to_list() == [[_A[0], _B[0]], [_A[0]]]
    assert propagated.ext.storage().to_list() == [None, None]


def test_null_polygons_are_ignored() -> None:
    out = _added(_multi_polygons([_P]), _listed_polygons([None, _Q, [[None]]]))  # type: ignore[list-item]

    assert out.ext.storage().to_list() == [[_P, _Q]]


def test_null_rings_are_ignored_or_propagate() -> None:
    geometry = _polygons(_P, _P)
    parts = _listed_polygons([[None], _Q], [_Q, [_P[0], None]])  # type: ignore[list-item]

    ignored = _added(geometry, parts)
    propagated = _added(geometry, parts, null_behavior="propagate")

    assert ignored.ext.storage().to_list() == [[_P, _Q], [_P, _Q]]
    assert propagated.ext.storage().to_list() == [None, None]


def test_parts_aggregated_from_rows_with_nulls() -> None:
    roads = pl.DataFrame({"region": ["n", "n", "s", "s"]}).with_columns(
        _lines(_B, None, None, _C, name="road")
    )
    highways = pl.DataFrame({"region": ["n", "s"]}).with_columns(
        _lines(_A, _A, name="highway")
    )

    out = highways.join(
        roads.group_by("region", maintain_order=True).agg("road"), on="region"
    ).select(geo.add_parts("highway", "road"))

    assert out.to_series().ext.storage().to_list() == [[_A, _B], [_A, _C]]


def test_empty_parts_are_kept_or_skipped() -> None:
    geometry, parts = _multi_lines([[]]), _listed_lines([[], _B, []])

    kept = _added(geometry, parts)
    skipped = _added(geometry, parts, empty_behavior="skip")

    assert kept.ext.storage().to_list() == [[[], [], _B, []]]
    # The geometry's own empty part stays.
    assert skipped.ext.storage().to_list() == [[[], _B]]


def test_an_empty_single_part_is_skipped() -> None:
    out = _added(_lines(_A, _A), _lines([], _B, name="parts"), empty_behavior="skip")

    assert out.ext.storage().to_list() == [[_A], [_A, _B]]


def test_empty_points_are_skipped() -> None:
    out = _added(
        _points(_A[0]), _listed_points([_EMPTY_POINT, _B[0]]), empty_behavior="skip"
    )

    assert out.ext.storage().to_list() == [[_A[0], _B[0]]]


def test_empty_polygons_are_skipped() -> None:
    out = _added(_polygons(_P), _listed_polygons([[], _Q, [[]]]), empty_behavior="skip")

    assert out.ext.storage().to_list() == [[_P, _Q]]


def test_null_and_empty_parts_are_both_left_out() -> None:
    out = _added(
        _lines(_A),
        _listed_lines([None, _B, [], _C, None]),
        empty_behavior="skip",
    )

    assert out.ext.storage().to_list() == [[_A, _B, _C]]


def test_a_single_part_is_added_to_every_row() -> None:
    df = pl.DataFrame([_lines(_A, _B)])

    out = df.select(geo.add_parts("geometry", _lines(_C))).to_series()

    assert out.ext.storage().to_list() == [[_A, _C], [_B, _C]]


def test_a_single_geometry_gets_every_row_of_parts() -> None:
    df = pl.DataFrame([_lines(_B, _C, name="parts")])

    out = df.select(geo.add_parts(_lines(_A), "parts")).to_series()

    assert out.ext.storage().to_list() == [[_A, _B], [_A, _C]]


def test_a_single_list_of_parts_with_nulls_and_empties_is_added_to_every_row() -> None:
    df = pl.DataFrame([_lines(_A, _B)])
    parts = _listed_lines([None, _C, []])  # type: ignore[list-item]

    out = df.select(geo.add_parts("geometry", parts, empty_behavior="skip")).to_series()

    assert out.ext.storage().to_list() == [[_A, _C], [_B, _C]]


def test_rows_of_different_lengths_are_refused() -> None:
    df = pl.DataFrame([_lines(_A, _B)])

    with pytest.raises(ComputeError, match="2 rows but the parts have 3"):
        df.select(geo.add_parts("geometry", _lines(_A, _B, _C, name="parts")))


def test_sliced_columns() -> None:
    df = pl.DataFrame(
        [
            _multi_lines([_C], [_A], None, [_B]),
            _listed_lines([_A], [_B, _C], [_A], [None, _A]).alias("parts"),
        ]
    ).slice(1, 3)

    out = df.select(geo.add_parts("geometry", "parts")).to_series()

    assert out.ext.storage().to_list() == [[_A, _B, _C], None, [_B, _A]]


def test_sliced_columns_with_empty_parts_skipped() -> None:
    # The empty part before the slice shifts every part after it by one,
    # until the parts are trimmed to the slice.
    df = pl.DataFrame(
        [_lines(_A, _A), _listed_lines([[]], [_B, [], _C]).alias("parts")]
    ).slice(1, 1)

    out = df.select(geo.add_parts("geometry", "parts", empty_behavior="skip"))

    assert out.to_series().ext.storage().to_list() == [[_A, _B, _C]]


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

    out = df.lazy().select(geo.add_parts("geometry", "parts")).collect(engine=engine)

    assert out.to_series().ext.storage().to_list() == [[_A, _B], None, [_B, _C]]


def test_the_namespace_forwards() -> None:
    df = pl.DataFrame([_lines(_A), _listed_lines([_B]).alias("parts")])

    assert_frame_equal(
        df.select(pl.col("geometry").geo.add_parts("parts", empty_behavior="skip")),
        df.select(geo.add_parts("geometry", "parts", empty_behavior="skip")),
    )


def test_parts_of_another_dimension_are_refused() -> None:
    xyz = pl.Series(
        "parts",
        [[{"x": 1.0, "y": 1.0, "z": 0.0}]],
        dtype=pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})),
    ).ext.to(LineStringXYZ())
    lf = pl.LazyFrame([_lines(_A), xyz]).select(geo.add_parts("geometry", "parts"))

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
        (_lines(_A), _multi_points(_A, name="parts")),
        (_polygons(_P), _of_points([_A], 1)),
        (_lines(_A), _of_points([[[_A]]], 3)),
        (_lines(_A), _bare([[[_A]]], 3)),
        (
            _polygons(_P),
            _nested([[[_P]]], pl.List(pl.List(pl.List(_XY))), MultiPolygonXY(), 1),
        ),
    ],
)
def test_parts_of_another_kind_are_refused(
    geometry: pl.Series, parts: pl.Series
) -> None:
    lf = pl.LazyFrame([geometry, parts]).select(geo.add_parts("geometry", "parts"))

    with pytest.raises(ComputeError, match="expected the parts as"):
        lf.collect_schema()


@pytest.mark.parametrize(
    "kwargs", [{"null_behavior": "drop"}, {"empty_behavior": "ignore"}]
)
def test_unknown_behaviors_are_refused(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        geo.add_parts("geometry", "parts", **kwargs)  # type: ignore[arg-type]
