"""The intersection of two geometries, forwarded to GEOS."""

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
    LineStringXYZ,
    MultiLineStringXY,
    MultiPointXY,
    MultiPointXYZ,
    MultiPolygonXY,
    PointXY,
    PointXYM,
    PointXYZ,
    PolygonXY,
    PolygonXYZ,
    PolygonXYZM,
)

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


def _intersection(
    a: tuple[GeoArrowType, list[str | None]],
    b: tuple[GeoArrowType, list[str | None]],
    **kwargs: Any,
) -> pl.Series:
    df = pl.DataFrame([_wkt("a", *a), _wkt("b", *b)])
    return df.select(geo.intersection("a", "b", **kwargs))["a"]


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
            close = tolerance > 0 and (
                shapely.hausdorff_distance(got_row, want, densify=0.1) <= tolerance
            )
            assert shapely.equals(got_row, want) or close, (
                f"row {row}: {got_row} != {want}"
            )


_SQUARE = "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"
# Two arms, at 0..1 and 2..3 in x, joined along the bottom.
_U = "POLYGON ((0 0, 3 0, 3 3, 2 3, 2 1, 1 1, 1 3, 0 3, 0 0))"
# Across both arms of `_U`, above where they join.
_ACROSS_THE_ARMS = "POLYGON ((-1 2, 4 2, 4 4, -1 4, -1 2))"
_ZIGZAG = "LINESTRING (0 0, 1 1, 2 0, 3 1, 4 0)"
_BELOW_HALF = "POLYGON ((-1 -1, 5 -1, 5 0.5, -1 0.5, -1 -1))"


# The kinds of the output, by input kinds: without `allow_splitting`, then with it.
# Only the upper half, as the table is symmetric.
_KINDS = (
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
)
_OUTPUT_KINDS = {
    ("Point", "Point"): ("Point", "Point"),
    ("Point", "MultiPoint"): ("Point", "Point"),
    ("Point", "LineString"): ("Point", "Point"),
    ("Point", "MultiLineString"): ("Point", "Point"),
    ("Point", "Polygon"): ("Point", "Point"),
    ("Point", "MultiPolygon"): ("Point", "Point"),
    ("MultiPoint", "MultiPoint"): ("MultiPoint", "MultiPoint"),
    ("MultiPoint", "LineString"): ("MultiPoint", "MultiPoint"),
    ("MultiPoint", "MultiLineString"): ("MultiPoint", "MultiPoint"),
    ("MultiPoint", "Polygon"): ("MultiPoint", "MultiPoint"),
    ("MultiPoint", "MultiPolygon"): ("MultiPoint", "MultiPoint"),
    ("LineString", "LineString"): ("LineString", "MultiLineString"),
    ("LineString", "MultiLineString"): ("LineString", "MultiLineString"),
    ("LineString", "Polygon"): ("LineString", "MultiLineString"),
    ("LineString", "MultiPolygon"): ("LineString", "MultiLineString"),
    ("MultiLineString", "MultiLineString"): ("MultiLineString", "MultiLineString"),
    ("MultiLineString", "Polygon"): ("MultiLineString", "MultiLineString"),
    ("MultiLineString", "MultiPolygon"): ("MultiLineString", "MultiLineString"),
    ("Polygon", "Polygon"): ("Polygon", "MultiPolygon"),
    ("Polygon", "MultiPolygon"): ("Polygon", "MultiPolygon"),
    ("MultiPolygon", "MultiPolygon"): ("MultiPolygon", "MultiPolygon"),
}
_DIMENSION = {"Point": 0, "LineString": 1, "Polygon": 2}
_KEEP = ("points", "lines", "polygons")
_SINGLE_OF_KEEP = ("Point", "LineString", "Polygon")


def _dimension(kind: str) -> int:
    return _DIMENSION[kind.removeprefix("Multi")]


def _expected_kind(a_kind: str, b_kind: str, keep: str | None, splits: bool) -> str:
    pair = (a_kind, b_kind) if (a_kind, b_kind) in _OUTPUT_KINDS else (b_kind, a_kind)
    kind = _OUTPUT_KINDS[pair][splits]
    if keep is None:
        return kind
    single = _SINGLE_OF_KEEP[_KEEP.index(keep)]
    return single if kind == kind.removeprefix("Multi") else f"Multi{single}"


def _output_dtype(a_kind: str, b_kind: str, **kwargs: Any) -> pl.DataType:
    a = pl.Series("a", [], dtype=_ORACLE_DTYPES[a_kind])
    b = pl.Series("b", [], dtype=_ORACLE_DTYPES[b_kind])
    lf = pl.LazyFrame([a, b]).select(geo.intersection("a", "b", **kwargs))
    return lf.collect_schema()["a"]


@pytest.mark.parametrize("allow_splitting", [False, True])
@pytest.mark.parametrize("b_kind", _KINDS)
@pytest.mark.parametrize("a_kind", _KINDS)
def test_the_output_kind_follows_the_table(
    a_kind: str, b_kind: str, allow_splitting: bool
) -> None:
    lowest = min(_dimension(a_kind), _dimension(b_kind))
    for keep in [None, *_KEEP[: lowest + 1]]:
        dtype = _output_dtype(
            a_kind, b_kind, keep=keep, allow_splitting=allow_splitting
        )
        expected = _expected_kind(a_kind, b_kind, keep, allow_splitting)
        assert type(dtype).__name__ == f"{expected}XY", f"keep={keep}"


@pytest.mark.parametrize("b_kind", _KINDS)
@pytest.mark.parametrize("a_kind", _KINDS)
def test_keeping_a_higher_dimension_than_the_inputs_is_refused(
    a_kind: str, b_kind: str
) -> None:
    lowest = min(_dimension(a_kind), _dimension(b_kind))
    for keep in _KEEP[lowest + 1 :]:
        with pytest.raises(ValueError, match=f"has no {keep}"):
            _output_dtype(a_kind, b_kind, keep=keep)


def test_an_unknown_keep_is_refused() -> None:
    with pytest.raises(ValueError, match="`keep` must be one of"):
        geo.intersection("a", "b", keep="areas")  # type: ignore[arg-type]


def test_overlapping_polygons_give_their_overlap() -> None:
    out = _intersection(
        (PolygonXY(), [_SQUARE, _SQUARE]),
        (PolygonXY(), ["POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))", _SQUARE]),
    )

    assert out.dtype == PolygonXY()
    _assert_geometries(out, ["POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))", _SQUARE])


def test_polygons_that_do_not_meet_give_an_empty_polygon() -> None:
    far = "POLYGON ((5 5, 6 5, 6 6, 5 5))"
    out = _intersection((PolygonXY(), [_SQUARE]), (PolygonXY(), [far]))

    assert _to_wkt(out) == ["POLYGON EMPTY"]


def test_polygons_touching_along_an_edge_give_the_edge_only_when_kept() -> None:
    a = (PolygonXY(), [_SQUARE])
    b = (PolygonXY(), ["POLYGON ((2 0, 3 0, 3 2, 2 2, 2 0))"])

    assert _to_wkt(_intersection(a, b)) == ["POLYGON EMPTY"]
    out = _intersection(a, b, keep="lines")
    assert out.dtype == LineStringXY()
    _assert_geometries(out, ["LINESTRING (2 0, 2 2)"])
    assert _to_wkt(_intersection(a, b, keep="points")) == ["POINT EMPTY"]


def test_a_dropped_part_is_not_a_split() -> None:
    # Overlaps `_SQUARE` in a square, and only touches it at (2, 0).
    touching = (
        "MULTIPOLYGON (((1 1, 3 1, 3 3, 1 3, 1 1)), ((2 -1, 3 -1, 3 0, 2 0, 2 -1)))"
    )

    out = _intersection((PolygonXY(), [_SQUARE]), (MultiPolygonXY(), [touching]))

    assert out.dtype == PolygonXY()
    _assert_geometries(out, ["POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))"])
    out = _intersection(
        (PolygonXY(), [_SQUARE]), (MultiPolygonXY(), [touching]), keep="points"
    )
    _assert_geometries(out, ["POINT (2 0)"])


# Touches `LINESTRING (0 0, 4 0)` at (1, 0) and (3, 0).
_TOUCHING_TWICE = "LINESTRING (0 1, 1 0, 2 1, 3 0, 4 1)"

# A pair that splits on each dimension, with the number of parts.
_SPLITS = {
    "polygons": ((PolygonXY(), _U), (PolygonXY(), _ACROSS_THE_ARMS), None, 2),
    "lines": ((LineStringXY(), _ZIGZAG), (PolygonXY(), _BELOW_HALF), None, 3),
    "points": (
        (LineStringXY(), "LINESTRING (0 0, 4 0)"),
        (LineStringXY(), _TOUCHING_TWICE),
        "points",
        2,
    ),
}


@pytest.mark.parametrize("dimension", _SPLITS)
def test_a_split_raises_without_allow_splitting(dimension: str) -> None:
    a, b, keep, parts = _SPLITS[dimension]

    with pytest.raises(ComputeError, match=f"split into {parts} parts"):
        _intersection((a[0], [a[1]]), (b[0], [b[1]]), keep=keep)


# A pair on each dimension that does not split, with what it gives.
_NO_SPLITS = {
    "polygons": (_SQUARE, _SQUARE, _SQUARE),
    "lines": ("LINESTRING (0 0, 4 0)", _BELOW_HALF, "LINESTRING (0 0, 4 0)"),
    "points": ("LINESTRING (0 0, 4 4)", "LINESTRING (0 4, 4 0)", "POINT (2 2)"),
}


@pytest.mark.parametrize("dimension", _SPLITS)
def test_a_split_gives_null_with_ignore_per_row_errors(dimension: str) -> None:
    a, b, keep, _ = _SPLITS[dimension]
    a_whole, b_whole, whole = _NO_SPLITS[dimension]

    out = _intersection(
        (a[0], [a[1], a_whole]),
        (b[0], [b[1], b_whole]),
        keep=keep,
        ignore_per_row_errors=True,
    )

    _assert_geometries(out, [None, whole])


def test_a_split_gives_every_part_with_allow_splitting() -> None:
    out = _intersection(
        (PolygonXY(), [_U, _SQUARE]),
        (PolygonXY(), [_ACROSS_THE_ARMS, _SQUARE]),
        allow_splitting=True,
    )

    assert out.dtype == MultiPolygonXY()
    arms = "MULTIPOLYGON (((0 2, 1 2, 1 3, 0 3, 0 2)), ((2 2, 3 2, 3 3, 2 3, 2 2)))"
    _assert_geometries(out, [arms, f"MULTIPOLYGON ({_SQUARE[8:]})"])


def test_polygons_touching_along_two_edges_split_their_lines() -> None:
    # Sits in the notch of `_U`, touching both arms but not the bottom.
    a = (PolygonXY(), [_U])
    b = (PolygonXY(), ["POLYGON ((1 2, 2 2, 2 3, 1 3, 1 2))"])

    with pytest.raises(ComputeError, match="split into 2 parts"):
        _intersection(a, b, keep="lines")
    out = _intersection(a, b, keep="lines", allow_splitting=True)
    assert out.dtype == MultiLineStringXY()
    _assert_geometries(out, ["MULTILINESTRING ((1 2, 1 3), (2 2, 2 3))"])


def test_a_line_is_clipped_into_its_pieces_inside_a_polygon() -> None:
    out = _intersection(
        (LineStringXY(), [_ZIGZAG]), (PolygonXY(), [_BELOW_HALF]), allow_splitting=True
    )

    assert out.dtype == MultiLineStringXY()
    pieces = "MULTILINESTRING ((0 0, 0.5 0.5), (1.5 0.5, 2 0, 2.5 0.5), (3.5 0.5, 4 0))"
    _assert_geometries(out, [pieces])


def test_a_line_crossing_itself_splits_even_inside_a_polygon() -> None:
    loop = "LINESTRING (0 0, 2 0, 2 1, 1 1, 1 -1)"
    big = "POLYGON ((-1 -2, 5 -2, 5 5, -1 5, -1 -2))"

    with pytest.raises(ComputeError, match="split into 3 parts"):
        _intersection((LineStringXY(), [loop]), (PolygonXY(), [big]))


def test_crossing_lines_give_their_crossings_when_points_are_kept() -> None:
    a = (LineStringXY(), ["LINESTRING (0 0, 4 4)", "LINESTRING (0 0, 4 0)"])
    b = (LineStringXY(), ["LINESTRING (0 4, 4 0)", _TOUCHING_TWICE])

    assert _to_wkt(_intersection(a, b)) == ["LINESTRING EMPTY", "LINESTRING EMPTY"]
    out = _intersection(a, b, keep="points", ignore_per_row_errors=True)
    assert out.dtype == PointXY()
    _assert_geometries(out, ["POINT (2 2)", None])
    out = _intersection(a, b, keep="points", allow_splitting=True)
    assert out.dtype == MultiPointXY()
    _assert_geometries(out, ["MULTIPOINT ((2 2))", "MULTIPOINT ((1 0), (3 0))"])


def test_overlapping_lines_give_their_overlap() -> None:
    out = _intersection(
        (LineStringXY(), ["LINESTRING (0 0, 4 0)"]),
        (LineStringXY(), ["LINESTRING (1 0, 2 0)"]),
    )

    _assert_geometries(out, ["LINESTRING (1 0, 2 0)"])


def test_a_point_gives_itself_inside_and_on_the_boundary_and_empty_outside() -> None:
    points = ["POINT (0.5 0.5)", "POINT (2 1)", "POINT (5 5)"]

    out = _intersection((PointXY(), points), (PolygonXY(), [_SQUARE] * 3))

    assert out.dtype == PointXY()
    assert _to_wkt(out) == ["POINT(0.5 0.5)", "POINT(2 1)", "POINT EMPTY"]


def test_a_multipoint_keeps_the_points_inside() -> None:
    out = _intersection(
        (
            MultiPointXY(),
            ["MULTIPOINT ((1 1), (5 5), (0.5 0.5))", None, "MULTIPOINT ((5 5))"],
        ),
        (PolygonXY(), [_SQUARE] * 3),
    )

    assert out.dtype == MultiPointXY()
    _assert_geometries(out, ["MULTIPOINT ((1 1), (0.5 0.5))", None, "MULTIPOINT EMPTY"])


def test_the_exterior_runs_counter_clockwise_with_ccw_outer() -> None:
    a = (PolygonXY(), [_SQUARE])
    b = (PolygonXY(), ["POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])

    def is_ccw(out: pl.Series) -> bool:
        return pl.select(geo.is_ccw(out)).item()

    assert not is_ccw(_intersection(a, b))
    assert is_ccw(_intersection(a, b, ccw_outer=True))


def test_ccw_outer_turns_every_part_and_keeps_holes_clockwise() -> None:
    def is_ccw(out: pl.Series) -> bool:
        return pl.select(geo.is_ccw(out)).item()

    a, b = (PolygonXY(), [_U]), (PolygonXY(), [_ACROSS_THE_ARMS])
    assert not is_ccw(_intersection(a, b, allow_splitting=True))
    split = _intersection(a, b, allow_splitting=True, ccw_outer=True)
    assert split.dtype == MultiPolygonXY()
    assert is_ccw(split)
    holed = (
        PolygonXY(),
        ["POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 3, 3 3, 3 1, 1 1))"],
    )
    big = (PolygonXY(), ["POLYGON ((-1 -1, 5 -1, 5 5, -1 5, -1 -1))"])
    assert not is_ccw(_intersection(holed, big))
    assert is_ccw(_intersection(holed, big, ccw_outer=True))


@pytest.mark.parametrize("ccw_outer", [False, True])
def test_a_hole_is_kept(ccw_outer: bool) -> None:
    holed = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 3, 3 3, 3 1, 1 1))"
    big = "POLYGON ((-1 -1, 5 -1, 5 5, -1 5, -1 -1))"

    out = _intersection(
        (PolygonXY(), [holed]), (PolygonXY(), [big]), ccw_outer=ccw_outer
    )

    _assert_geometries(out, [holed])


@pytest.mark.parametrize("z_first", [True, False])
def test_z_from_one_side_is_filled_in(z_first: bool) -> None:
    with_z = (LineStringXYZ(), ["LINESTRING Z (0 0 10, 2 2 30)"])
    without = (LineStringXY(), ["LINESTRING (0 2, 2 0)"])
    a, b = (with_z, without) if z_first else (without, with_z)

    out = _intersection(a, b, keep="points")

    assert out.dtype == PointXYZ()
    assert _to_wkt(out) == ["POINT Z(1 1 20)"]


def test_z_from_one_side_is_filled_in_for_a_multipoint() -> None:
    flat = "POLYGON Z ((0 0 7, 2 0 7, 2 2 7, 0 2 7, 0 0 7))"
    a = (MultiPointXY(), ["MULTIPOINT ((1 1), (0.5 0.5), (9 9))"])

    out = _intersection(a, (PolygonXYZ(), [flat]))

    assert out.dtype == MultiPointXYZ()
    got = shapely.from_wkt(_to_wkt(out)[0])
    assert shapely.equals(got, shapely.from_wkt("MULTIPOINT ((1 1), (0.5 0.5))"))
    assert shapely.get_coordinates(got, include_z=True)[:, 2].tolist() == [7.0, 7.0]


def test_polygons_with_z_give_polygons_with_z() -> None:
    with_z = "POLYGON Z ((0 0 1, 2 0 1, 2 2 1, 0 2 1, 0 0 1))"
    b = (PolygonXY(), ["POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])

    out = _intersection((PolygonXYZ(), [with_z]), b)

    assert out.dtype == PolygonXYZ()
    assert shapely.equals(
        shapely.from_wkt(_to_wkt(out)[0]),
        shapely.from_wkt("POLYGON Z ((1 1 1, 2 1 1, 2 2 1, 1 2 1, 1 1 1))"),
    )
    assert (
        shapely.get_coordinates(shapely.from_wkt(_to_wkt(out)[0]), include_z=True)[
            :, 2
        ].tolist()
        == [1.0] * 5
    )


_ZIGZAG_Z = "LINESTRING Z (0 0 7, 1 1 7, 2 0 7, 3 1 7, 4 0 7)"
_U_Z = "POLYGON Z ((0 0 7, 3 0 7, 3 3 7, 2 3 7, 2 1 7, 1 1 7, 1 3 7, 0 3 7, 0 0 7))"


@pytest.mark.parametrize(
    ("a", "b", "allow_splitting", "expected"),
    [
        (
            (MultiPointXYZ(), ["MULTIPOINT Z ((1 1 7), (9 9 7))"]),
            (PolygonXY(), [_SQUARE]),
            False,
            "MULTIPOINT Z ((1 1 7))",
        ),
        (
            (LineStringXYZ(), ["LINESTRING Z (-1 1 7, 3 1 7)"]),
            (PolygonXY(), [_SQUARE]),
            False,
            "LINESTRING Z (0 1 7, 2 1 7)",
        ),
        (
            (LineStringXYZ(), [_ZIGZAG_Z]),
            (PolygonXY(), [_BELOW_HALF]),
            True,
            (
                "MULTILINESTRING Z ((0 0 7, 0.5 0.5 7), (1.5 0.5 7, 2 0 7, 2.5 0.5 7),"
                " (3.5 0.5 7, 4 0 7))"
            ),
        ),
        (
            (PolygonXYZ(), [_U_Z]),
            (PolygonXY(), [_ACROSS_THE_ARMS]),
            True,
            (
                "MULTIPOLYGON Z (((0 2 7, 1 2 7, 1 3 7, 0 3 7, 0 2 7)),"
                " ((2 2 7, 3 2 7, 3 3 7, 2 3 7, 2 2 7)))"
            ),
        ),
    ],
    ids=["multipoint", "linestring", "multilinestring", "multipolygon"],
)
def test_z_is_kept_for_every_list_kind(
    a: tuple[GeoArrowType, list[str | None]],
    b: tuple[GeoArrowType, list[str | None]],
    allow_splitting: bool,
    expected: str,
) -> None:
    out = _intersection(a, b, allow_splitting=allow_splitting)

    assert type(out.dtype).__name__.endswith("XYZ")
    got = shapely.from_wkt(_to_wkt(out)[0])
    assert shapely.equals(got, shapely.from_wkt(expected))
    assert set(shapely.get_coordinates(got, include_z=True)[:, 2].tolist()) == {7.0}


def test_m_is_dropped() -> None:
    a = (PointXYM(), ["POINT M (0.5 0.5 3)"])
    out = _intersection(a, (PolygonXY(), [_SQUARE]))
    assert out.dtype == PointXY()
    assert _to_wkt(out) == ["POINT(0.5 0.5)"]

    # The m differs from the z on every vertex, so neither can pass for the other.
    a = (
        PolygonXYZM(),
        ["POLYGON ZM ((0 0 1 10, 2 0 1 20, 2 2 1 30, 0 2 1 40, 0 0 1 10))"],
    )
    out = _intersection(a, (LineStringXYM(), ["LINESTRING M (-1 1 5, 3 1 5)"]))
    assert out.dtype == LineStringXYZ()
    got = shapely.from_wkt(_to_wkt(out)[0])
    assert shapely.equals(got, shapely.from_wkt("LINESTRING (0 1, 2 1)"))
    assert shapely.get_coordinates(got, include_z=True)[:, 2].tolist() == [1.0, 1.0]


def test_an_empty_geometry_on_either_side_gives_an_empty_one() -> None:
    a = (PolygonXY(), ["POLYGON EMPTY", _SQUARE])
    b = (PolygonXY(), [_SQUARE, "POLYGON EMPTY"])

    assert _to_wkt(_intersection(a, b)) == ["POLYGON EMPTY", "POLYGON EMPTY"]
    empty_point = {"x": math.nan, "y": math.nan}
    df = pl.DataFrame(
        [_series("a", PointXY(), [empty_point]), _wkt("b", PolygonXY(), [_SQUARE])]
    )
    assert _to_wkt(df.select(geo.intersection("a", "b"))["a"]) == ["POINT EMPTY"]


def test_a_null_geometry_on_either_side_gives_a_null_result() -> None:
    out = _intersection(
        (PolygonXY(), [None, _SQUARE, None]), (PolygonXY(), [_SQUARE, None, None])
    )

    assert out.to_list() == [None, None, None]


def test_columns_chunked_differently_are_intersected_row_by_row() -> None:
    line = "LINESTRING (0 0, 1 0, 1 1)"
    a = _chunked(
        [
            _wkt("a", LineStringXY(), [line]),
            _wkt("a", LineStringXY(), [None, line, line]),
        ]
    )
    far = "LINESTRING (5 5, 6 6)"
    b = _chunked(
        [
            _wkt("b", LineStringXY(), [line, line, far]),
            _wkt("b", LineStringXY(), [None]),
        ]
    )

    out = pl.DataFrame([a, b]).select(geo.intersection("a", "b"))["a"]

    _assert_geometries(out, [line, None, "LINESTRING EMPTY", None])


def test_a_single_geometry_is_intersected_with_every_row() -> None:
    many = _chunked(
        [
            _wkt("many", LineStringXY(), ["LINESTRING (-1 1, 3 1)", None]),
            _wkt("many", LineStringXY(), ["LINESTRING (5 5, 6 6)"]),
        ]
    )
    one = _wkt("one", PolygonXY(), [_SQUARE])
    expected = ["LINESTRING (0 1, 2 1)", None, "LINESTRING EMPTY"]
    df = pl.DataFrame(many)

    out = df.select(geo.intersection("many", one))["many"]
    assert out.dtype == LineStringXY()
    _assert_geometries(out, expected)
    out = df.select(geo.intersection(one, "many")).to_series()
    assert out.dtype == LineStringXY()
    _assert_geometries(out, expected)


def test_a_single_geometry_keeps_its_side() -> None:
    line = _wkt("line", LineStringXY(), ["LINESTRING (0 0, 4 0)"])
    many = pl.DataFrame(_wkt("many", LineStringXY(), ["LINESTRING (3 0, 1 0)"] * 2))

    # GEOS gives the overlap in the direction of the first line.
    first = many.select(geo.intersection(line, "many")).to_series()
    second = many.select(geo.intersection("many", line))["many"]
    assert _to_wkt(first) == ["LINESTRING(1 0,3 0)"] * 2
    assert _to_wkt(second) == ["LINESTRING(3 0,1 0)"] * 2


# GEOS fails to build a linestring of a single vertex.
_ONE_VERTEX_LINE = _xy((0, 0))
# GEOS raises on this one, but not on every geometry that is not finite.
_NOT_FINITE_LINE = _xy((0, 0), (math.nan, 1), (2, 2))
_FINITE_LINE = _xy((0, 0), (1, 0), (1, 1))


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_a_row_paired_with_a_null_never_reaches_geos(early_finite_check: bool) -> None:
    df = pl.DataFrame(
        [
            _series("a", LineStringXY(), [None, None, _FINITE_LINE]),
            _series(
                "b", LineStringXY(), [_ONE_VERTEX_LINE, _NOT_FINITE_LINE, _FINITE_LINE]
            ),
        ]
    )

    out = df.select(geo.intersection("a", "b", early_finite_check=early_finite_check))

    _assert_geometries(out["a"], [None, None, "LINESTRING (0 0, 1 0, 1 1)"])


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_against_a_single_null_geometry_no_row_reaches_geos(
    early_finite_check: bool,
) -> None:
    many = _series("many", LineStringXY(), [_ONE_VERTEX_LINE, _NOT_FINITE_LINE])
    one = _wkt("one", PolygonXY(), [None])
    df = pl.DataFrame(many)
    kwargs = {"early_finite_check": early_finite_check}

    out = df.select(geo.intersection("many", one, **kwargs))["many"]
    assert out.to_list() == [None, None]
    assert out.dtype == LineStringXY()
    out = df.select(geo.intersection(one, "many", **kwargs)).to_series()
    assert out.to_list() == [None, None]


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_a_row_paired_with_an_empty_geometry_never_reaches_geos(
    early_finite_check: bool,
) -> None:
    around = "POLYGON ((-1 -1, 3 -1, 3 3, -1 3, -1 -1))"
    df = pl.DataFrame(
        [
            _series(
                "a", LineStringXY(), [_ONE_VERTEX_LINE, _NOT_FINITE_LINE, _FINITE_LINE]
            ),
            _wkt("b", PolygonXY(), ["POLYGON EMPTY", "POLYGON EMPTY", around]),
        ]
    )
    expected = ["LINESTRING EMPTY", "LINESTRING EMPTY", "LINESTRING (0 0, 1 0, 1 1)"]
    kwargs = {"early_finite_check": early_finite_check}

    _assert_geometries(df.select(geo.intersection("a", "b", **kwargs))["a"], expected)
    _assert_geometries(df.select(geo.intersection("b", "a", **kwargs))["b"], expected)


@pytest.mark.parametrize("early_finite_check", [False, True])
@pytest.mark.parametrize(
    ("one", "empty"),
    [
        (_wkt("one", PolygonXY(), ["POLYGON EMPTY"]), "LINESTRING EMPTY"),
        (_series("one", PointXY(), [{"x": math.nan, "y": math.nan}]), "POINT EMPTY"),
    ],
    ids=["polygon", "point"],
)
def test_against_a_single_empty_geometry_no_row_reaches_geos(
    one: pl.Series, empty: str, early_finite_check: bool
) -> None:
    many = _series("many", LineStringXY(), [_ONE_VERTEX_LINE, _NOT_FINITE_LINE])
    df = pl.DataFrame(many)
    kwargs = {"early_finite_check": early_finite_check}

    out = df.select(geo.intersection("many", one, **kwargs))["many"]
    assert _to_wkt(out) == [empty, empty]
    out = df.select(geo.intersection(one, "many", **kwargs)).to_series()
    assert _to_wkt(out) == [empty, empty]


def _not_finite_lines(**kwargs: bool) -> pl.Series:
    """A line that is not finite, then one that is, each intersected with a square around both."""
    df = pl.DataFrame(
        [
            _series("a", LineStringXY(), [_NOT_FINITE_LINE, _FINITE_LINE]),
            _wkt("b", PolygonXY(), ["POLYGON ((-1 -1, 3 -1, 3 3, -1 3, -1 -1))"] * 2),
        ]
    )
    return df.select(geo.intersection("a", "b", **kwargs))["a"]


def test_geos_raises_on_a_line_that_is_not_finite() -> None:
    with pytest.raises(ComputeError, match="NaN/Inf"):
        _not_finite_lines()


def test_geos_failing_on_a_row_gives_null_with_ignore_per_row_errors() -> None:
    out = _not_finite_lines(ignore_per_row_errors=True)

    _assert_geometries(out, [None, "LINESTRING (0 0, 1 0, 1 1)"])


def test_the_early_finite_check_raises_before_geos() -> None:
    with pytest.raises(ComputeError, match="GEOS expects finite x and y"):
        _not_finite_lines(early_finite_check=True)
    out = _not_finite_lines(early_finite_check=True, ignore_per_row_errors=True)
    _assert_geometries(out, [None, "LINESTRING (0 0, 1 0, 1 1)"])


@pytest.mark.parametrize("single_first", [True, False])
def test_ignore_per_row_errors_with_a_single_geometry(single_first: bool) -> None:
    def run(one: pl.Series, many: pl.Series, **kwargs: bool) -> pl.Series:
        a, b = (one, "many") if single_first else ("many", one)
        return pl.DataFrame(many).select(geo.intersection(a, b, **kwargs)).to_series()

    big = _wkt("one", PolygonXY(), ["POLYGON ((-1 -1, 3 -1, 3 3, -1 3, -1 -1))"])
    lines = _series("many", LineStringXY(), [_NOT_FINITE_LINE, _FINITE_LINE])
    out = run(big, lines, ignore_per_row_errors=True)
    _assert_geometries(out, [None, "LINESTRING (0 0, 1 0, 1 1)"])

    bad = _series("one", LineStringXY(), [_NOT_FINITE_LINE])
    squares = _wkt("many", PolygonXY(), [_SQUARE, _SQUARE])
    with pytest.raises(ComputeError, match="NaN/Inf"):
        run(bad, squares)
    assert run(bad, squares, ignore_per_row_errors=True).to_list() == [None, None]


def test_the_namespace_method_passes_its_options_on() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_U]), _wkt("b", PolygonXY(), [_ACROSS_THE_ARMS])]
    )

    out = df.select(pl.col("a").geo.intersection("b", ignore_per_row_errors=True))
    assert out["a"].to_list() == [None]
    out = df.select(
        pl.col("a").geo.intersection(
            "b", keep="lines", allow_splitting=True, ccw_outer=True
        )
    )
    assert out["a"].dtype == MultiLineStringXY()


def test_the_crs_of_a_is_kept() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (1 1)"])]
    ).with_columns(geo.set_crs("a", "EPSG:3857"), geo.set_crs("b", "EPSG:3857"))

    out = df.select(geo.intersection("a", "b"))

    assert out["a"].dtype._crs() == df["a"].dtype._crs()


def test_different_crss_are_refused() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"), geo.set_crs("b", "EPSG:3857"))

    with pytest.raises(ValueError, match="different CRSs"):
        df.select(geo.intersection("a", "b"))


def test_a_crs_on_only_one_side_is_refused() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"))

    with pytest.raises(ValueError, match="only one of the geometries"):
        df.select(geo.intersection("a", "b"))


def test_unequal_lengths_are_refused() -> None:
    a = _wkt("a", PointXY(), ["POINT (0 0)", "POINT (1 1)"])
    b = _wkt("b", PolygonXY(), [_SQUARE, _SQUARE, _SQUARE])

    with pytest.raises((ShapeError, ComputeError)):
        pl.select(geo.intersection(a, b))


_ORACLE_DTYPES = {
    "Point": PointXY(),
    "MultiPoint": MultiPointXY(),
    "LineString": LineStringXY(),
    "MultiLineString": MultiLineStringXY(),
    "Polygon": PolygonXY(),
    "MultiPolygon": MultiPolygonXY(),
}


def _random_shape(rng: random.Random, kind: str) -> Any:
    """A small shape on a grid, so that shapes often meet, touch and overlap.
    Polygons include a U-shape, so that their intersections can split.
    The parts of a multipolygon are far apart, so that it stays valid.
    """

    def vertex() -> tuple[int, int]:
        return rng.randint(0, 4), rng.randint(0, 4)

    def line() -> Any:
        return shapely.LineString([vertex() for _ in range(rng.randint(2, 4))])

    def polygon() -> Any:
        if rng.random() < 0.3:
            return shapely.affinity.translate(
                shapely.from_wkt(_U), rng.randint(0, 2), rng.randint(0, 2)
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


def _kept_parts(intersection: Any, dimension: int) -> list[Any]:
    parts = shapely.get_parts(intersection)
    return [
        p for p in parts if not p.is_empty and shapely.get_dimensions(p) == dimension
    ]


@pytest.mark.parametrize("keep", [None, *_KEEP])
@pytest.mark.parametrize("b_kind", _ORACLE_DTYPES)
@pytest.mark.parametrize("a_kind", _ORACLE_DTYPES)
def test_matches_shapely(a_kind: str, b_kind: str, keep: str | None) -> None:
    lowest = min(_dimension(a_kind), _dimension(b_kind))
    dimension = lowest if keep is None else _KEEP.index(keep)
    if dimension > lowest:
        pytest.skip(f"{a_kind} and {b_kind} cannot keep {keep}")
    rng = random.Random(f"{a_kind} {b_kind}")
    a = [_random_shape(rng, a_kind) for _ in range(200)]
    b = [_random_shape(rng, b_kind) for _ in range(200)]
    df = pl.DataFrame(
        [
            _wkt("a", _ORACLE_DTYPES[a_kind], list(shapely.to_wkt(a))),
            _wkt("b", _ORACLE_DTYPES[b_kind], list(shapely.to_wkt(b))),
        ]
    )
    kept = [_kept_parts(shapely.intersection(x, y), dimension) for x, y in zip(a, b)]

    def intersections(**kwargs: bool) -> pl.Series:
        out = df.select(
            geo.intersection("a", "b", keep=keep, ignore_per_row_errors=True, **kwargs)
        )
        return out["a"]

    every_part = intersections(allow_splitting=True)
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
    if _expected_kind(a_kind, b_kind, keep, splits=False).startswith("Multi"):
        return
    # The shapes are made to split in the dimension of the inputs, and a point cannot split.
    if keep is None and "Point" not in (a_kind, b_kind):
        assert any(len(parts) > 1 for parts in kept), "some rows should split"
    expected = [
        None if shapely.get_num_geometries(shapely.from_wkt(wkt)) > 1 else wkt
        for wkt in _to_wkt(every_part)
    ]
    _assert_geometries(intersections(), expected)
