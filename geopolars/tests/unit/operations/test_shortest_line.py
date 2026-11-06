"""The shortest line between two geometries, forwarded to GEOS."""

from __future__ import annotations

import math
import random
from typing import Any

import polars as pl
import pytest
import shapely
import shapely.affinity
from polars.exceptions import ComputeError, ShapeError

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    MultiPolygonXYZM,
    PointXY,
    PointXYZ,
    PolygonXY,
)


def _wkt(name: str, dtype: GeoArrowType, wkts: list[str | None]) -> pl.Series:
    out = pl.select(geo.from_wkt(pl.Series(wkts, dtype=pl.String), type(dtype)))
    return out.to_series().alias(name)


def _chunked(series: list[pl.Series]) -> pl.Series:
    out = pl.concat(series, rechunk=False)
    assert out.n_chunks() == len(series)
    return out


def _as_wkt(lines: pl.Series) -> list[str | None]:
    return pl.select(geo.to_wkt(lines)).to_series().to_list()


def _shortest_lines(
    a: tuple[GeoArrowType, list[str | None]],
    b: tuple[GeoArrowType, list[str | None]],
    **kwargs: bool,
) -> list[str | None]:
    df = pl.DataFrame([_wkt("a", *a), _wkt("b", *b)])
    return _as_wkt(df.select(geo.shortest_line("a", "b", **kwargs))["a"])


_SQUARE = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"


def test_the_line_runs_from_a_to_b() -> None:
    lines = _shortest_lines(
        (PointXY(), ["POINT (3 0.5)"]),
        (MultiPolygonXY(), [f"MULTIPOLYGON ({_SQUARE[7:]})"]),
    )

    assert lines == ["LINESTRING(3 0.5,1 0.5)"]


def test_overlapping_geometries_give_a_line_of_length_zero() -> None:
    lines = _shortest_lines((PointXY(), ["POINT (0.5 0.5)"]), (PolygonXY(), [_SQUARE]))

    assert lines == ["LINESTRING(0.5 0.5,0.5 0.5)"]


def test_z_and_m_are_dropped() -> None:
    a = _wkt("a", PointXYZ(), ["POINT Z (3 0.5 7)"])
    b = _wkt(
        "b",
        MultiPolygonXYZM(),
        ["MULTIPOLYGON ZM (((0 0 1 2, 1 0 1 2, 1 1 1 2, 0 1 1 2, 0 0 1 2)))"],
    )

    out = pl.DataFrame([a, b]).select(geo.shortest_line("a", "b"))

    assert out.schema["a"] == LineStringXY()
    assert _as_wkt(out["a"]) == ["LINESTRING(3 0.5,1 0.5)"]


def test_an_empty_geometry_on_either_side_gives_an_empty_line() -> None:
    a = (PointXY(), ["POINT EMPTY", "POINT (0 0)"])
    b = (MultiPolygonXY(), ["MULTIPOLYGON EMPTY", "MULTIPOLYGON EMPTY"])
    expected = ["LINESTRING EMPTY", "LINESTRING EMPTY"]

    assert _shortest_lines(a, b) == expected
    assert _shortest_lines(b, a) == expected


def test_a_null_geometry_on_either_side_gives_a_null_line() -> None:
    a = (PolygonXY(), [_SQUARE, None, None])
    b = (PointXY(), [None, "POINT (0 0)", None])

    assert _shortest_lines(a, b) == [None, None, None]


def test_a_single_geometry_is_measured_against_every_row() -> None:
    many = _chunked(
        [
            _wkt("many", PointXY(), ["POINT (3 0)", None]),
            _wkt("many", PointXY(), ["POINT (0 3)", "POINT EMPTY"]),
        ]
    )
    one = _wkt("one", PolygonXY(), [_SQUARE])
    df = pl.DataFrame(many)

    out = df.select(geo.shortest_line(one, "many"))
    assert _as_wkt(out.to_series()) == [
        "LINESTRING(1 0,3 0)",
        None,
        "LINESTRING(0 1,0 3)",
        "LINESTRING EMPTY",
    ]
    out = df.select(geo.shortest_line("many", one))
    assert _as_wkt(out["many"]) == [
        "LINESTRING(3 0,1 0)",
        None,
        "LINESTRING(0 3,0 1)",
        "LINESTRING EMPTY",
    ]


# GEOS fails on a vertex with just one NaN axis, at least when it is the closest one.
_NOT_FINITE_POINT = pl.Series("a", [{"x": math.nan, "y": 1.0}]).ext.to(PointXY())


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_against_a_single_null_geometry_no_row_reaches_geos(
    early_finite_check: bool,
) -> None:
    one = _wkt("one", PolygonXY(), [None])
    kwargs = {"early_finite_check": early_finite_check}

    out = pl.select(geo.shortest_line(_NOT_FINITE_POINT, one, **kwargs))
    assert out.to_series().to_list() == [None]
    assert out.schema["a"] == LineStringXY()
    out = pl.select(geo.shortest_line(one, _NOT_FINITE_POINT, **kwargs))
    assert out.to_series().to_list() == [None]


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_a_row_paired_with_a_null_never_reaches_geos(early_finite_check: bool) -> None:
    b = _wkt("b", PolygonXY(), [None])

    out = pl.select(
        geo.shortest_line(_NOT_FINITE_POINT, b, early_finite_check=early_finite_check)
    )

    assert out.to_series().to_list() == [None]


def test_geos_raises_on_a_geometry_that_is_not_finite() -> None:
    b = _wkt("b", PolygonXY(), [_SQUARE])

    with pytest.raises(ComputeError, match="GEOSNearestPoints_r failed"):
        pl.select(geo.shortest_line(_NOT_FINITE_POINT, b))


def test_the_early_finite_check_raises_before_geos() -> None:
    b = _wkt("b", PolygonXY(), [_SQUARE])

    with pytest.raises(ComputeError, match="GEOS expects finite x and y"):
        pl.select(geo.shortest_line(_NOT_FINITE_POINT, b, early_finite_check=True))


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_a_failing_row_gives_null_with_ignore_per_row_errors(
    early_finite_check: bool,
) -> None:
    a = pl.concat([_NOT_FINITE_POINT, _wkt("a", PointXY(), ["POINT (3 0)"])])
    df = pl.DataFrame([a, _wkt("b", PolygonXY(), [_SQUARE, _SQUARE])])

    out = df.select(
        pl.col("a").geo.shortest_line(
            "b", ignore_per_row_errors=True, early_finite_check=early_finite_check
        )
    )

    assert _as_wkt(out["a"]) == [None, "LINESTRING(3 0,1 0)"]


def test_the_line_carries_the_crs() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (3 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:28992"), geo.set_crs("b", "EPSG:28992"))

    out = df.select(geo.shortest_line("a", "b"))

    assert out.schema["a"] == LineStringXY(crs="EPSG:28992")


def test_different_crss_are_refused() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"), geo.set_crs("b", "EPSG:3857"))

    with pytest.raises(ValueError, match="different CRSs"):
        df.select(geo.shortest_line("a", "b"))


def test_unequal_lengths_are_refused() -> None:
    a = _wkt("a", PointXY(), ["POINT (0 0)", "POINT (1 1)"])
    b = _wkt("b", PolygonXY(), [_SQUARE, _SQUARE, _SQUARE])

    with pytest.raises((ShapeError, ComputeError)):
        pl.select(geo.shortest_line(a, b))


_ORACLE_DTYPES = {
    "Point": PointXY(),
    "MultiPoint": MultiPointXY(),
    "LineString": LineStringXY(),
    "MultiLineString": MultiLineStringXY(),
    "Polygon": PolygonXY(),
    "MultiPolygon": MultiPolygonXY(),
}


def _random_shape(rng: random.Random, kind: str) -> Any:
    def vertices(n: int) -> list[tuple[float, float]]:
        return [(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(n)]

    def polygon() -> Any:
        return shapely.convex_hull(shapely.MultiPoint(vertices(5)))

    if rng.random() < 0.05:
        return shapely.from_wkt(f"{kind.upper()} EMPTY")
    match kind:
        case "Point":
            return shapely.Point(vertices(1))
        case "MultiPoint":
            return shapely.MultiPoint(vertices(3))
        case "LineString":
            return shapely.LineString(vertices(3))
        case "MultiLineString":
            return shapely.MultiLineString([vertices(2), vertices(3)])
        case "Polygon":
            return polygon()
        case _:
            far = shapely.affinity.translate(polygon(), 20)
            return shapely.MultiPolygon([polygon(), far])


@pytest.mark.parametrize("b_kind", _ORACLE_DTYPES)
@pytest.mark.parametrize("a_kind", _ORACLE_DTYPES)
def test_matches_shapely(a_kind: str, b_kind: str) -> None:
    rng = random.Random(f"{a_kind} {b_kind}")
    a = [_random_shape(rng, a_kind) for _ in range(100)]
    b = [_random_shape(rng, b_kind) for _ in range(100)]
    df = pl.DataFrame(
        [
            _wkt("a", _ORACLE_DTYPES[a_kind], list(shapely.to_wkt(a, rounding_precision=-1))),
            _wkt("b", _ORACLE_DTYPES[b_kind], list(shapely.to_wkt(b, rounding_precision=-1))),
        ]
    )

    out = shapely.from_wkt(_as_wkt(df.select(geo.shortest_line("a", "b"))["a"]))

    expected = shapely.shortest_line(a, b)
    expected = [shapely.LineString() if e is None else e for e in expected]
    assert all(shapely.equals_exact(out, expected, tolerance=1e-12))
