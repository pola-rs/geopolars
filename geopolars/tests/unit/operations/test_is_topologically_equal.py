"""Whether two geometries are topologically equal, forwarded to GEOS."""

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


_FINITE_LINE = _xy((0, 0), (1, 0), (1, 1))


def _wkt(name: str, dtype: GeoArrowType, wkts: list[str | None]) -> pl.Series:
    out = pl.select(geo.from_wkt(pl.Series(wkts, dtype=pl.String), type(dtype)))
    return out.to_series().alias(name)


def _topologically_equal(
    a: tuple[GeoArrowType, list[str | None]],
    b: tuple[GeoArrowType, list[str | None]],
    **kwargs: bool,
) -> list[bool | None]:
    df = pl.DataFrame([_wkt("a", *a), _wkt("b", *b)])
    out = df.select(geo.is_topologically_equal("a", "b", **kwargs))
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

    out = df.select(geo.is_topologically_equal("a", "b"))

    assert out["a"].to_list() == [True, False]


def test_an_empty_geometry_is_not_equal_to_one_with_coordinates() -> None:
    a = (LineStringXY(), ["LINESTRING EMPTY", "LINESTRING (0 0, 1 1)"])
    b = (MultiLineStringXY(), ["MULTILINESTRING ((0 0, 1 1))", "MULTILINESTRING EMPTY"])

    assert _topologically_equal(a, b) == [False, False]


def test_a_null_geometry_on_either_side_gives_a_null_result() -> None:
    a = (PolygonXY(), [_SQUARE, None, None])
    b = (MultiPolygonXY(), [None, f"MULTIPOLYGON ({_SQUARE[7:]})", None])

    assert _topologically_equal(a, b) == [None, None, None]


def test_a_single_geometry_is_compared_against_every_row() -> None:
    many = _wkt(
        "many",
        PolygonXY(),
        [_SQUARE, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))", "POLYGON EMPTY", None],
    )
    one = _wkt("one", MultiPolygonXY(), [f"MULTIPOLYGON ({_SQUARE[7:]})"])
    df = pl.DataFrame(many)
    expected = [True, True, False, None]

    out = df.select(geo.is_topologically_equal("many", one))
    assert out["many"].to_list() == expected
    out = df.select(geo.is_topologically_equal(one, "many"))
    assert out.to_series().to_list() == expected


def test_against_a_single_null_geometry_everything_is_null() -> None:
    many = _wkt("many", PolygonXY(), [_SQUARE, "POLYGON EMPTY"])
    one = _wkt("one", PointXY(), [None])

    out = pl.DataFrame(many).select(geo.is_topologically_equal("many", one))

    assert out["many"].to_list() == [None, None]


def test_columns_chunked_differently_are_compared_row_by_row() -> None:
    a = _chunked([_wkt("a", PolygonXY(), [_SQUARE]), _wkt("a", PolygonXY(), [None])])
    b = _wkt("b", MultiPolygonXY(), [f"MULTIPOLYGON ({_SQUARE[7:]})", None])

    out = pl.DataFrame([a, b]).select(geo.is_topologically_equal("a", "b"))

    assert out["a"].to_list() == [True, None]


def test_a_chunked_column_is_compared_against_a_single_geometry() -> None:
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

    out = df.select(geo.is_topologically_equal("many", one))
    assert out["many"].to_list() == expected
    out = df.select(geo.is_topologically_equal(one, "many"))
    assert out.to_series().to_list() == expected


def test_m_is_not_compared() -> None:
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
    rows = [_NOT_FINITE_LINE, _FINITE_LINE]
    df = pl.DataFrame(
        [_series("a", LineStringXY(), rows), _series("b", LineStringXY(), rows)]
    )
    return df.select(geo.is_topologically_equal("a", "b", **kwargs))["a"].to_list()


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


def test_the_namespace_method_passes_its_options_on() -> None:
    df = pl.DataFrame(
        [
            _series("a", LineStringXY(), [_NOT_FINITE_LINE, _FINITE_LINE]),
            _series("b", LineStringXY(), [_NOT_FINITE_LINE, _FINITE_LINE[::-1]]),
        ]
    )

    out = df.select(
        pl.col("a").geo.is_topologically_equal("b", ignore_per_row_errors=True)
    )

    assert out["a"].to_list() == [None, True]


def test_a_point_that_is_not_finite_is_compared_as_it_is() -> None:
    a = [
        {"x": math.nan, "y": 1.0},
        {"x": math.inf, "y": 1.0},
        {"x": math.nan, "y": 1.0},
    ]
    b = [{"x": math.nan, "y": 1.0}, {"x": math.inf, "y": 1.0}, {"x": 0.0, "y": 1.0}]
    df = pl.DataFrame([_series("a", PointXY(), a), _series("b", PointXY(), b)])

    out = df.select(geo.is_topologically_equal("a", "b"))

    assert out["a"].to_list() == [True, True, False]


def test_different_crss_are_refused() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"), geo.set_crs("b", "EPSG:3857"))

    with pytest.raises(ValueError, match="different CRSs"):
        df.select(geo.is_topologically_equal("a", "b"))


def test_unequal_lengths_are_refused() -> None:
    a = _wkt("a", PointXY(), ["POINT (0 0)", "POINT (1 1)"])
    b = _wkt("b", PolygonXY(), [_SQUARE, _SQUARE, _SQUARE])

    with pytest.raises((ShapeError, ComputeError)):
        pl.select(geo.is_topologically_equal(a, b))


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
def test_matches_shapely(a_kind: str, b_kind: str) -> None:
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

    out = df.select(geo.is_topologically_equal("a", "b"))["a"].to_list()

    expected = list(shapely.equals(a, b))
    assert out == expected
    assert any(expected), "the cases should not all be unequal"
