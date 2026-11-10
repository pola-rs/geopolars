"""Whether two geometries intersect, forwarded to GEOS."""

from __future__ import annotations

import math
import random
from typing import Any

import polars as pl
import pytest
import shapely
from polars.exceptions import ComputeError

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYM,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PolygonXY,
    PolygonXYZ,
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


def _intersects(
    a: tuple[GeoArrowType, list[str | None]],
    b: tuple[GeoArrowType, list[str | None]],
    **kwargs: bool,
) -> list[bool | None]:
    df = pl.DataFrame([_wkt("a", *a), _wkt("b", *b)])
    out = df.select(geo.intersects("a", "b", **kwargs))
    assert out.schema == pl.Schema({"a": pl.Boolean})
    return out["a"].to_list()


_SQUARE = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
_NEIGHBOUR = "POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))"
_FAR_SQUARE = "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))"
_LARGE_SQUARE = "POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))"


def test_overlapping_touching_and_disjoint_polygons() -> None:
    a = (PolygonXY(), [_SQUARE, _SQUARE, _SQUARE])
    b = (PolygonXY(), [_LARGE_SQUARE, _NEIGHBOUR, _FAR_SQUARE])

    assert _intersects(a, b) == [True, True, False]


def test_z_and_m_are_not_compared() -> None:
    a = "POLYGON Z ((5 5 0, 6 5 0, 6 6 0, 5 6 0, 5 5 0))"
    line = "LINESTRING M (0 0 1, 1 1 2)"

    assert _intersects((PolygonXYZ(), [a]), (PolygonXY(), [_FAR_SQUARE])) == [True]
    assert _intersects((LineStringXYM(), [line]), (PolygonXY(), [_SQUARE])) == [True]


@pytest.mark.parametrize(
    ("dtype", "wkt"),
    [
        (PointXY(), "POINT (0.5 0.5)"),
        (LineStringXY(), "LINESTRING (0 0, 1 1)"),
        (PolygonXY(), _SQUARE),
    ],
    ids=["point", "linestring", "polygon"],
)
@pytest.mark.parametrize(
    ("empty_dtype", "empty_wkt"),
    [
        (PolygonXY(), "POLYGON EMPTY"),
        (MultiPointXY(), "MULTIPOINT EMPTY"),
        (LineStringXY(), "LINESTRING EMPTY"),
    ],
    ids=["polygon", "multipoint", "linestring"],
)
def test_an_empty_geometry_intersects_nothing(
    dtype: GeoArrowType, wkt: str, empty_dtype: GeoArrowType, empty_wkt: str
) -> None:
    assert _intersects((empty_dtype, [empty_wkt]), (dtype, [wkt])) == [False]
    assert _intersects((dtype, [wkt]), (empty_dtype, [empty_wkt])) == [False]


def test_two_empty_geometries() -> None:
    empty_point = {"x": math.nan, "y": math.nan}
    df = pl.DataFrame(
        [
            _series("a", PointXY(), [empty_point]),
            _wkt("b", PolygonXY(), ["POLYGON EMPTY"]),
        ]
    )

    out = df.select(geo.intersects("a", "b"))

    assert out["a"].to_list() == [False]


def test_a_null_geometry_on_either_side_gives_a_null_result() -> None:
    a = (PolygonXY(), [_SQUARE, None, None, "POLYGON EMPTY"])
    b = (MultiPolygonXY(), [None, f"MULTIPOLYGON ({_SQUARE[7:]})", None, None])

    assert _intersects(a, b) == [None, None, None, None]


def test_a_single_geometry_is_tested_against_every_row() -> None:
    many = _wkt(
        "many",
        PolygonXY(),
        [_SQUARE, _NEIGHBOUR, _FAR_SQUARE, "POLYGON EMPTY", None],
    )
    one = _wkt("one", LineStringXY(), ["LINESTRING (-1 0.5, 0.5 0.5)"])
    df = pl.DataFrame(many)
    expected = [True, False, False, False, None]

    assert df.select(geo.intersects(one, "many")).to_series().to_list() == expected
    assert df.select(geo.intersects("many", one))["many"].to_list() == expected


def test_a_single_empty_geometry_intersects_no_row() -> None:
    many = _wkt("many", PolygonXY(), [_SQUARE, None])
    one = _wkt("one", PolygonXY(), ["POLYGON EMPTY"])

    out = pl.DataFrame(many).select(geo.intersects("many", one))

    assert out["many"].to_list() == [False, None]


def test_against_a_single_null_geometry_everything_is_null() -> None:
    many = _wkt("many", PolygonXY(), [_SQUARE, "POLYGON EMPTY"])
    one = _wkt("one", PointXY(), [None])

    out = pl.DataFrame(many).select(geo.intersects("many", one))

    assert out["many"].to_list() == [None, None]


def test_columns_chunked_differently_are_tested_row_by_row() -> None:
    a = _chunked([_wkt("a", PolygonXY(), [_SQUARE]), _wkt("a", PolygonXY(), [None])])
    b = _wkt("b", PolygonXY(), [_LARGE_SQUARE, _SQUARE])

    out = pl.DataFrame([a, b]).select(geo.intersects("a", "b"))

    assert out["a"].to_list() == [True, None]


def test_a_chunked_column_is_tested_against_a_single_geometry() -> None:
    many = _chunked(
        [
            _wkt("many", PolygonXY(), [_SQUARE]),
            _wkt("many", PolygonXY(), [None, _FAR_SQUARE]),
        ]
    )
    one = _wkt("one", PolygonXY(), [_NEIGHBOUR])

    out = pl.DataFrame(many).select(geo.intersects("many", one))

    assert out["many"].to_list() == [True, None, False]


# GEOS raises on this one, but not on every geometry that is not finite.
_NOT_FINITE_LINE = _xy((0, 0), (math.nan, 1), (2, 2))
_FINITE_LINE = _xy((0, 0), (1, 0), (1, 1))


def _not_finite_lines(**kwargs: bool) -> list[bool | None]:
    """A line that is not finite, then one that is, each tested against itself."""
    rows = [_NOT_FINITE_LINE, _FINITE_LINE]
    df = pl.DataFrame(
        [_series("a", LineStringXY(), rows), _series("b", LineStringXY(), rows)]
    )
    return df.select(geo.intersects("a", "b", **kwargs))["a"].to_list()


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
    rows = [_NOT_FINITE_LINE, _FINITE_LINE]
    df = pl.DataFrame(
        [_series("a", LineStringXY(), rows), _series("b", LineStringXY(), rows)]
    )

    out = df.select(pl.col("a").geo.intersects("b", ignore_per_row_errors=True))

    assert out["a"].to_list() == [None, True]


def test_different_crss_are_refused() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"), geo.set_crs("b", "EPSG:3857"))

    with pytest.raises(ValueError, match="different CRSs"):
        df.select(geo.intersects("a", "b"))


_ORACLE_DTYPES = {
    "Point": PointXY(),
    "MultiPoint": MultiPointXY(),
    "LineString": LineStringXY(),
    "MultiLineString": MultiLineStringXY(),
    "Polygon": PolygonXY(),
    "MultiPolygon": MultiPolygonXY(),
}


def _random_shape(rng: random.Random, kind: str) -> Any:
    """A shape of `kind` on a small grid, or sometimes an empty one.
    The grid is small so that shapes often touch or share vertices.
    """
    if rng.random() < 0.1:
        return shapely.from_wkt(f"{kind.upper()} EMPTY")

    def vertex() -> tuple[int, int]:
        return (rng.randint(0, 3), rng.randint(0, 3))

    def polygon() -> Any:
        x, y = vertex()
        width, height = rng.randint(1, 2), rng.randint(1, 2)
        return shapely.box(x, y, x + width, y + height)

    match kind:
        case "Point":
            return shapely.Point(vertex())
        case "MultiPoint":
            return shapely.MultiPoint([vertex() for _ in range(rng.randint(1, 3))])
        case "LineString":
            return shapely.LineString([vertex(), vertex(), vertex()])
        case "MultiLineString":
            return shapely.MultiLineString([[vertex(), vertex()] for _ in range(2)])
        case "Polygon":
            return polygon()
        case _:
            return shapely.MultiPolygon([polygon(), shapely.box(10, 10, 11, 11)])


def _oracle_columns(a_kind: str, b_kind: str) -> tuple[list[Any], list[Any]]:
    rng = random.Random(f"{a_kind} {b_kind}")
    a = [_random_shape(rng, a_kind) for _ in range(200)]
    b = [_random_shape(rng, b_kind) for _ in range(200)]
    return a, b


@pytest.mark.parametrize("b_kind", _ORACLE_DTYPES)
@pytest.mark.parametrize("a_kind", _ORACLE_DTYPES)
def test_matches_shapely(a_kind: str, b_kind: str) -> None:
    a, b = _oracle_columns(a_kind, b_kind)
    df = pl.DataFrame(
        [
            _wkt("a", _ORACLE_DTYPES[a_kind], list(shapely.to_wkt(a))),
            _wkt("b", _ORACLE_DTYPES[b_kind], list(shapely.to_wkt(b))),
        ]
    )

    out = df.select(geo.intersects("a", "b"))["a"].to_list()

    assert out == list(shapely.intersects(a, b))


@pytest.mark.parametrize("b_kind", _ORACLE_DTYPES)
@pytest.mark.parametrize("a_kind", _ORACLE_DTYPES)
def test_a_single_geometry_matches_shapely(a_kind: str, b_kind: str) -> None:
    a, b = _oracle_columns(a_kind, b_kind)
    one = next(shape for shape in a if not shape.is_empty)
    one_series = _wkt("one", _ORACLE_DTYPES[a_kind], [shapely.to_wkt(one)])
    df = pl.DataFrame(_wkt("b", _ORACLE_DTYPES[b_kind], list(shapely.to_wkt(b))))

    one_against_many = df.select(geo.intersects(one_series, "b")).to_series()
    many_against_one = df.select(geo.intersects("b", one_series)).to_series()

    # Some of the lines have zero length, on which GEOS answers differently once prepared.
    shapely.prepare(one)
    expected = list(shapely.intersects(one, b))
    assert one_against_many.to_list() == expected
    assert many_against_one.to_list() == expected
