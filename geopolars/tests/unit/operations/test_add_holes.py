"""Adding holes to polygons from another column."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    LineStringXYZ,
    MultiLineStringXY,
    PolygonXY,
)

RD = "EPSG:28992"

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

Ring = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Ring:
    return [{"x": x, "y": y} for x, y in xy]


_EXTERIOR = _ring((0, 0), (10, 0), (0, 10), (0, 0))
_HOLE = _ring((1, 1), (2, 1), (1, 2), (1, 1))
_OTHER_HOLE = _ring((3, 3), (4, 3), (3, 4), (3, 3))
_THIRD_HOLE = _ring((5, 1), (6, 1), (5, 2), (5, 1))
_OPEN = _ring((1, 1), (2, 1), (1, 2))


def _polygons(*polygons: list[Ring] | None, crs: str | None = None) -> pl.Series:
    storage = pl.Series("polygon", list(polygons), dtype=pl.List(_XY_VERTICES))
    return storage.ext.to(PolygonXY(crs=crs))


def _lines(*lines: Ring | None, crs: str | None = None) -> pl.Series:
    storage = pl.Series("holes", list(lines), dtype=_XY_VERTICES)
    return storage.ext.to(LineStringXY(crs=crs))


def _listed_lines(*rows: list[Ring | None] | None, crs: str | None = None) -> pl.Series:
    storage = pl.Series("holes", list(rows), dtype=pl.List(_XY_VERTICES))
    return storage.list.eval(pl.element().ext.to(LineStringXY(crs=crs)))


def _added(polygons: pl.Series, holes: pl.Series, **kwargs: object) -> pl.Series:
    df = pl.DataFrame([polygons, holes])
    return df.select(geo.add_holes("polygon", "holes", **kwargs)).to_series()


def test_a_single_linestring_is_added_as_a_hole() -> None:
    out = _added(_polygons([_EXTERIOR], crs=RD), _lines(_HOLE, crs=RD))

    assert out.dtype == PolygonXY(crs=RD)
    assert out.ext.storage().to_list() == [[_EXTERIOR, _HOLE]]


def test_a_list_is_added_after_the_existing_holes() -> None:
    out = _added(
        _polygons([_EXTERIOR, _HOLE], [_EXTERIOR]),
        _listed_lines([_OTHER_HOLE, _THIRD_HOLE], [_HOLE]),
    )

    assert out.ext.storage().to_list() == [
        [_EXTERIOR, _HOLE, _OTHER_HOLE, _THIRD_HOLE],
        [_EXTERIOR, _HOLE],
    ]


def test_a_multilinestring_is_added_like_a_list() -> None:
    rows = ([_OTHER_HOLE, _THIRD_HOLE], [], None)
    polygons = _polygons([_EXTERIOR, _HOLE], [_EXTERIOR], [_EXTERIOR])
    multi = pl.Series("holes", rows, dtype=pl.List(_XY_VERTICES))

    out = _added(polygons, multi.ext.to(MultiLineStringXY()))

    assert_frame_equal(
        out.to_frame(), _added(polygons, _listed_lines(*rows)).to_frame()
    )


def test_nulls() -> None:
    out = _added(
        _polygons(None, [_EXTERIOR], [_EXTERIOR], [_EXTERIOR]),
        _listed_lines([_HOLE], None, [], [_HOLE, None]),
    )

    assert out.ext.storage().to_list() == [None, [_EXTERIOR], [_EXTERIOR], None]


def test_an_empty_polygon_has_no_room_for_holes() -> None:
    out = _added(_polygons([], [], []), _listed_lines([_HOLE], [], None))

    assert out.ext.storage().to_list() == [None, [], []]


def test_an_open_hole_nulls_the_polygon_unless_closed() -> None:
    polygons, holes = _polygons([_EXTERIOR]), _lines(_OPEN)

    checked = _added(polygons, holes)
    closed = _added(polygons, holes, close=geo.CloseRings())

    assert checked.is_null().to_list() == [True]
    assert closed.ext.storage().to_list() == [[_EXTERIOR, _HOLE]]


def test_the_namespace_forwards() -> None:
    df = pl.DataFrame([_polygons([_EXTERIOR]), _lines(_HOLE)])

    assert_frame_equal(
        df.select(pl.col("polygon").geo.add_holes("holes")),
        df.select(geo.add_holes("polygon", "holes")),
    )


def test_holes_of_another_dimension_are_refused() -> None:
    xyz = pl.Series(
        "holes",
        [[{"x": 1.0, "y": 1.0, "z": 0.0}]],
        dtype=pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})),
    ).ext.to(LineStringXYZ())
    lf = pl.LazyFrame([_polygons([_EXTERIOR]), xyz]).select(
        geo.add_holes("polygon", "holes")
    )

    with pytest.raises(ComputeError, match="the polygon is xy but the holes are xyz"):
        lf.collect_schema()


def test_holes_in_another_crs_are_refused() -> None:
    lf = pl.LazyFrame(
        [_polygons([_EXTERIOR], crs=RD), _lines(_HOLE, crs="EPSG:4326")]
    ).select(geo.add_holes("polygon", "holes"))

    with pytest.raises(ComputeError, match="different CRSs"):
        lf.collect_schema()


def test_a_non_polygon_is_refused() -> None:
    df = pl.DataFrame([_lines(_EXTERIOR).alias("polygon"), _lines(_HOLE)])

    with pytest.raises(TypeError, match="add_holes expects a polygon column"):
        df.select(geo.add_holes("polygon", "holes"))
