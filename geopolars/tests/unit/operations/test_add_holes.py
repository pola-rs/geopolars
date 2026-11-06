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
    MultiPointXY,
    PolygonXY,
)
from tests.unit.conftest import Dimension

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


def test_a_single_ring_is_added_to_every_polygon() -> None:
    df = pl.DataFrame([_polygons([_EXTERIOR], [_EXTERIOR])])

    out = df.select(geo.add_holes("polygon", _lines(_HOLE))).to_series()

    assert out.ext.storage().to_list() == [[_EXTERIOR, _HOLE], [_EXTERIOR, _HOLE]]


def test_a_single_polygon_gets_every_row_of_holes() -> None:
    df = pl.DataFrame([_lines(_HOLE, _OTHER_HOLE)])

    out = df.select(geo.add_holes(_polygons([_EXTERIOR]), "holes")).to_series()

    assert out.ext.storage().to_list() == [
        [_EXTERIOR, _HOLE],
        [_EXTERIOR, _OTHER_HOLE],
    ]


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


def test_adding_the_holes_back_rebuilds_the_polygons(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    exteriors = dimension.polygons(ring_coords.filter(pl.col("ring") == 0))
    holes = (
        ring_coords.group_by("polygon", "ring", maintain_order=True)
        .agg(dimension.point())
        .select("polygon", "ring", dimension.linestring())
        .group_by("polygon", maintain_order=True)
        .agg(pl.col("line").filter(pl.col("ring") > 0).alias("holes"))
    )

    out = pl.concat([exteriors, holes.select("holes")], how="horizontal").select(
        geo.add_holes("polygon", "holes")
    )

    assert_frame_equal(out, dimension.polygons(ring_coords))


@pytest.mark.parametrize(
    ("null_parts", "expected"),
    [
        ("null", [None, None, [_EXTERIOR], None, None]),
        ("drop", [None, [_EXTERIOR], [_EXTERIOR], [_EXTERIOR, _HOLE], [_EXTERIOR]]),
    ],
)
def test_nulls(null_parts: str, expected: list[object]) -> None:
    out = _added(
        _polygons(None, [_EXTERIOR], [_EXTERIOR], [_EXTERIOR], [_EXTERIOR]),
        _listed_lines([_HOLE], None, [], [_HOLE, None], [[*_HOLE[:3], None]]),
        null_parts=null_parts,
    )

    assert out.ext.storage().to_list() == expected


@pytest.mark.parametrize(
    ("null_parts", "expected"), [("null", [None]), ("drop", [[_EXTERIOR]])]
)
def test_an_untyped_null_is_a_null_row_of_holes(
    null_parts: str, expected: list[object]
) -> None:
    df = pl.DataFrame([_polygons([_EXTERIOR])])

    out = df.select(
        geo.add_holes("polygon", pl.lit(None), null_parts=null_parts)
    ).to_series()

    assert out.ext.storage().to_list() == expected


def test_holes_dropped_before_a_slice_stay_dropped() -> None:
    df = pl.DataFrame(
        [
            _polygons([_EXTERIOR], [_EXTERIOR]),
            _listed_lines([None, _OTHER_HOLE], [None, _HOLE, None, _THIRD_HOLE]),
        ]
    ).slice(1, 1)

    out = df.select(geo.add_holes("polygon", "holes", null_parts="drop")).to_series()

    assert out.ext.storage().to_list() == [[_EXTERIOR, _HOLE, _THIRD_HOLE]]


def test_sliced_columns() -> None:
    df = pl.DataFrame(
        [
            _polygons([_HOLE], [_EXTERIOR, _HOLE], None, [_EXTERIOR]),
            _listed_lines([_OTHER_HOLE], [_OTHER_HOLE], [_HOLE], [_THIRD_HOLE, _HOLE]),
        ]
    ).slice(1, 3)

    out = df.select(geo.add_holes("polygon", "holes")).to_series()

    assert out.ext.storage().to_list() == [
        [_EXTERIOR, _HOLE, _OTHER_HOLE],
        None,
        [_EXTERIOR, _THIRD_HOLE, _HOLE],
    ]


def test_an_empty_polygon_has_no_room_for_holes() -> None:
    out = _added(
        _polygons([], [], []), _listed_lines([_HOLE], [], None), null_parts="drop"
    )

    assert out.ext.storage().to_list() == [None, [], []]


def test_a_dropped_hole_takes_no_room_in_an_empty_polygon() -> None:
    out = _added(_polygons([]), _listed_lines([[*_HOLE[:3], None]]), null_parts="drop")

    assert out.ext.storage().to_list() == [[]]


def test_an_empty_hole_can_be_added_to_an_empty_polygon() -> None:
    out = _added(_polygons([], [[]]), _lines([], []))

    assert out.ext.storage().to_list() == [[[]], [[], []]]


def test_an_empty_exterior_has_no_room_for_holes() -> None:
    out = _added(_polygons([[]]), _lines(_HOLE))

    assert out.to_list() == [None]


def test_a_polygon_with_an_empty_exterior_can_be_kept() -> None:
    out = _added(
        _polygons([], [[]], [_EXTERIOR]),
        _lines(_HOLE, _HOLE, _HOLE),
        empty_exterior="keep",
    )

    assert out.ext.storage().to_list() == [[], [[]], [_EXTERIOR, _HOLE]]


def test_a_kept_polygon_takes_none_of_its_holes_unless_all_are_empty() -> None:
    out = _added(
        _polygons([[]], [[]]),
        _listed_lines([[], _HOLE, []], [[], []]),
        empty_exterior="keep",
    )

    assert out.ext.storage().to_list() == [[[]], [[], [], []]]


def test_a_kept_polygon_ignores_the_holes_that_are_dropped() -> None:
    broken = [*_HOLE[:3], None]
    out = _added(
        _polygons([[]], [[]]),
        _listed_lines([broken, []], [None, _HOLE]),  # type: ignore[list-item]
        null_parts="drop",
        empty_exterior="keep",
    )

    assert out.ext.storage().to_list() == [[[], []], [[]]]


def test_a_null_hole_still_nulls_a_kept_polygon() -> None:
    out = _added(_polygons([]), _lines(None), empty_exterior="keep")

    assert out.to_list() == [None]


def test_an_unknown_empty_exterior_is_refused() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        geo.add_holes("polygon", "holes", empty_exterior="drop")  # type: ignore[arg-type]


def test_an_open_hole_nulls_the_polygon_unless_closed() -> None:
    polygons, holes = _polygons([_EXTERIOR]), _lines(_OPEN)

    checked = _added(polygons, holes)
    closed = _added(polygons, holes, close=geo.CloseRings())

    assert checked.is_null().to_list() == [True]
    assert closed.ext.storage().to_list() == [[_EXTERIOR, _HOLE]]


def test_an_open_ring_already_in_the_polygon_is_checked_and_closed_too() -> None:
    polygons, holes = _polygons([_EXTERIOR, _OPEN]), _listed_lines([])

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


@pytest.mark.parametrize(
    ("polygon_crs", "holes", "match"),
    [
        (RD, _lines(_HOLE, crs="EPSG:4326"), "different CRSs"),
        (RD, _lines(_HOLE), "only one"),
        (None, _lines(_HOLE, crs=RD), "only one"),
        (RD, _listed_lines([_HOLE]), "only one"),
        (None, _listed_lines([_HOLE], crs=RD), "only one"),
    ],
)
def test_holes_in_another_crs_are_refused(
    polygon_crs: str | None, holes: pl.Series, match: str
) -> None:
    lf = pl.LazyFrame([_polygons([_EXTERIOR], crs=polygon_crs), holes]).select(
        geo.add_holes("polygon", "holes")
    )

    with pytest.raises(ComputeError, match=match):
        lf.collect_schema()


def test_holes_as_bare_vertices_go_with_any_crs() -> None:
    holes = pl.Series("holes", [_HOLE], dtype=_XY_VERTICES)

    out = _added(_polygons([_EXTERIOR], crs=RD), holes)

    assert out.dtype == PolygonXY(crs=RD)
    assert out.ext.storage().to_list() == [[_EXTERIOR, _HOLE]]


def test_holes_as_multipoints_are_accepted() -> None:
    multi_points = pl.Series("holes", [_HOLE], dtype=_XY_VERTICES).ext.to(
        MultiPointXY()
    )
    listed = multi_points.implode()

    single = _added(_polygons([_EXTERIOR]), multi_points)
    several = _added(_polygons([_EXTERIOR]), listed)

    assert single.ext.storage().to_list() == [[_EXTERIOR, _HOLE]]
    assert several.ext.storage().to_list() == [[_EXTERIOR, _HOLE]]


def test_holes_of_another_kind_are_refused() -> None:
    lf = pl.LazyFrame(
        [_polygons([_EXTERIOR]), _polygons([_HOLE]).alias("holes")]
    ).select(geo.add_holes("polygon", "holes"))

    with pytest.raises(ComputeError, match="expected the holes as one `linestring`"):
        lf.collect_schema()


def test_a_non_polygon_is_refused() -> None:
    df = pl.DataFrame([_lines(_EXTERIOR).alias("polygon"), _lines(_HOLE)])

    with pytest.raises(TypeError, match="add_holes expects a polygon column"):
        df.select(geo.add_holes("polygon", "holes"))
