"""The line spanning the Hausdorff distance between two geometries."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import polars as pl
import pytest
import shapely
import shapely.affinity
from polars.exceptions import ComputeError

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYZ,
    LineStringXYZM,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PointXYM,
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


def _hausdorff_lines(
    a: tuple[GeoArrowType, list[str | None]],
    b: tuple[GeoArrowType, list[str | None]],
    **kwargs: Any,
) -> list[str | None]:
    df = pl.DataFrame([_wkt("a", *a), _wkt("b", *b)])
    return _as_wkt(df.select(geo.hausdorff_line_between("a", "b", **kwargs))["a"])


def _coords_storage(axes: str) -> pl.Struct:
    return pl.Struct(dict.fromkeys(axes, pl.Float64))


def _line_strings(
    name: str, rows: list[list[tuple[float, ...]]], axes: str = "xy"
) -> pl.Series:
    coords = [[dict(zip(axes, vertex, strict=True)) for vertex in row] for row in rows]
    series = pl.Series(name, coords, dtype=pl.List(_coords_storage(axes)))
    dtype = {"xy": LineStringXY(), "xyz": LineStringXYZ()}[axes]
    return series.ext.to(dtype)


def _multi_points(name: str, rows: list[list[tuple[float, float]]]) -> pl.Series:
    coords = [[{"x": x, "y": y} for x, y in row] for row in rows]
    series = pl.Series(name, coords, dtype=pl.List(_coords_storage("xy")))
    return series.ext.to(MultiPointXY())


_SQUARE = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
_BUMP = "LINESTRING (0 1, 5 4, 10 1)"


def test_the_line_runs_from_a_to_b() -> None:
    line = (LineStringXY(), ["LINESTRING (0 0, 10 0)"])
    point = (PointXY(), ["POINT (0 1)"])

    assert _hausdorff_lines(line, point) == ["LINESTRING(10 0,0 1)"]
    assert _hausdorff_lines(point, line) == ["LINESTRING(0 1,10 0)"]


def test_the_line_can_end_between_two_vertices() -> None:
    lines = _hausdorff_lines(
        (LineStringXY(), ["LINESTRING (0 0, 10 0)"]), (LineStringXY(), [_BUMP])
    )

    assert lines == ["LINESTRING(5 0,5 4)"]


def test_a_polygon_is_measured_by_its_rings() -> None:
    corners_and_centre = "MULTIPOINT ((0 0), (1 0), (1 1), (0 1), (0.5 0.5))"

    lines = _hausdorff_lines(
        (MultiPointXY(), [corners_and_centre]), (PolygonXY(), [_SQUARE])
    )

    assert lines == ["LINESTRING(0.5 0.5,0.5 0)"]


def test_z_and_m_are_kept_and_interpolated_between_two_vertices() -> None:
    a = _wkt("a", LineStringXYZM(), ["LINESTRING ZM (0 0 0 20, 10 0 10 40)"])
    b = _wkt("b", LineStringXYZM(), ["LINESTRING ZM (0 1 7 1, 5 4 7 2, 10 1 7 3)"])

    out = pl.DataFrame([a, b]).select(geo.hausdorff_line_between("a", "b"))

    assert out.schema["a"] == LineStringXYZM()
    assert _as_wkt(out["a"]) == ["LINESTRING ZM(5 0 5 30,5 4 7 2)"]


@pytest.mark.parametrize(
    ("b_dtype", "axis", "expected"),
    [(PointXYZ(), "z", LineStringXYZ()), (PointXYM(), "m", LineStringXY())],
)
def test_the_line_has_the_axes_both_sides_have(
    b_dtype: GeoArrowType, axis: str, expected: GeoArrowType
) -> None:
    a = _wkt("a", LineStringXYZ(), ["LINESTRING Z (0 0 1, 10 0 2)"])
    b = pl.Series("b", [{"x": 0.0, "y": 1.0, axis: 3.0}]).ext.to(b_dtype)

    out = pl.DataFrame([a, b]).select(geo.hausdorff_line_between("a", "b"))

    assert out.schema["a"] == expected


def test_an_empty_geometry_on_either_side_gives_an_empty_line() -> None:
    a = (PointXY(), ["POINT EMPTY", "POINT (0 0)", "POINT (0 0)"])
    b = (PolygonXY(), [_SQUARE, "POLYGON EMPTY", "POLYGON (EMPTY)"])
    expected = ["LINESTRING EMPTY"] * 3

    assert _hausdorff_lines(a, b) == expected
    assert _hausdorff_lines(b, a) == expected


def test_a_null_geometry_on_either_side_gives_a_null_line() -> None:
    a = (PolygonXY(), [_SQUARE, None, None])
    b = (PointXY(), [None, "POINT (0 0)", None])

    assert _hausdorff_lines(a, b) == [None, None, None]


def test_a_single_vertex_line_string_counts_as_a_point() -> None:
    lines = _hausdorff_lines(
        (LineStringXY(), ["LINESTRING (3 0)"]), (PolygonXY(), [_SQUARE])
    )

    assert lines == ["LINESTRING(3 0,0 1)"]


def test_a_single_geometry_is_measured_against_every_row() -> None:
    many = _chunked(
        [
            _wkt("many", PointXY(), ["POINT (3 0)", None]),
            _wkt("many", PointXY(), ["POINT (0 3)", "POINT EMPTY"]),
        ]
    )
    one = _wkt("one", LineStringXY(), ["LINESTRING (0 0, 1 0)"])
    df = pl.DataFrame(many)

    out = df.select(geo.hausdorff_line_between(one, "many"))
    assert _as_wkt(out.to_series()) == [
        "LINESTRING(0 0,3 0)",
        None,
        "LINESTRING(1 0,0 3)",
        "LINESTRING EMPTY",
    ]
    out = df.select(geo.hausdorff_line_between("many", one))
    assert _as_wkt(out["many"]) == [
        "LINESTRING(3 0,0 0)",
        None,
        "LINESTRING(0 3,1 0)",
        "LINESTRING EMPTY",
    ]


@pytest.mark.parametrize(
    ("one", "expected"),
    [
        (_wkt("one", PolygonXY(), ["POLYGON EMPTY"]), ["LINESTRING EMPTY", None]),
        (_wkt("one", PointXY(), ["POINT EMPTY"]), ["LINESTRING EMPTY", None]),
        (_wkt("one", PointXY(), [None]), [None, None]),
    ],
)
def test_a_single_empty_or_null_geometry(
    one: pl.Series, expected: list[str | None]
) -> None:
    df = pl.DataFrame(_wkt("many", PointXY(), ["POINT (3 0)", None]))

    out = df.select(geo.hausdorff_line_between(one, "many"))
    assert _as_wkt(out.to_series()) == expected
    out = df.select(geo.hausdorff_line_between("many", one))
    assert _as_wkt(out["many"]) == expected


def test_columns_chunked_differently_are_paired_row_by_row() -> None:
    a = _chunked([_wkt("a", PointXY(), ["POINT (3 0)"]), _wkt("a", PointXY(), [None])])
    b = _wkt("b", PolygonXY(), [_SQUARE, _SQUARE])

    out = pl.DataFrame([a, b]).select(geo.hausdorff_line_between("a", "b"))

    assert _as_wkt(out["a"]) == ["LINESTRING(3 0,0 1)", None]


_BUMP_SERIES = _wkt("b", LineStringXY(), [_BUMP])


def _lines_with_vertex(vertex: tuple[float, float], **kwargs: Any) -> list[str | None]:
    a = _line_strings("a", [[(0, 0), vertex, (10, 0)]])
    out = pl.DataFrame([a, _BUMP_SERIES]).select(
        geo.hausdorff_line_between("a", "b", **kwargs)
    )
    return _as_wkt(out["a"])


@pytest.mark.parametrize("vertex", [(math.nan, 1.0), (math.inf, 1.0), (math.nan,) * 2])
def test_a_vertex_that_is_not_finite_raises_by_default(
    vertex: tuple[float, float],
) -> None:
    with pytest.raises(ComputeError, match="expects finite coordinates"):
        _lines_with_vertex(vertex)


@pytest.mark.parametrize("vertex", [(math.nan, 1.0), (math.inf, 1.0), (math.nan,) * 2])
def test_skip_runs_the_line_on_past_a_vertex_that_is_not_finite(
    vertex: tuple[float, float],
) -> None:
    assert _lines_with_vertex(vertex, non_finite="skip") == ["LINESTRING(5 0,5 4)"]


def test_skip_empty_skips_an_empty_point() -> None:
    lines = _lines_with_vertex((math.nan, math.nan), non_finite="skip_empty")

    assert lines == ["LINESTRING(5 0,5 4)"]


def test_skip_empty_raises_on_any_other_vertex_that_is_not_finite() -> None:
    with pytest.raises(ComputeError, match=r"expects finite coordinates, got \(NaN, 1"):
        _lines_with_vertex((math.nan, 1.0), non_finite="skip_empty")


@pytest.mark.parametrize("non_finite", ["raise", "skip_empty"])
def test_an_empty_point_inside_a_multipoint_raises_without_skip(
    non_finite: str,
) -> None:
    a = _multi_points("a", [[(0, 0), (math.nan, math.nan)]])
    df = pl.DataFrame([a, _wkt("b", PointXY(), ["POINT (0 1)"])])
    if non_finite == "raise":
        with pytest.raises(ComputeError, match="expects finite coordinates"):
            df.select(geo.hausdorff_line_between("a", "b", non_finite=non_finite))
    else:
        out = df.select(geo.hausdorff_line_between("a", "b", non_finite=non_finite))
        assert _as_wkt(out["a"]) == ["LINESTRING(0 0,0 1)"]


def test_a_geometry_left_without_vertices_gives_an_empty_line() -> None:
    a = _multi_points("a", [[(math.nan, math.nan), (math.nan, 1.0)]])
    df = pl.DataFrame([a, _wkt("b", PointXY(), ["POINT (0 1)"])])

    out = df.select(geo.hausdorff_line_between("a", "b", non_finite="skip"))

    assert _as_wkt(out["a"]) == ["LINESTRING EMPTY"]


def test_an_empty_point_row_never_raises() -> None:
    lines = _hausdorff_lines((PointXY(), ["POINT EMPTY"]), (PolygonXY(), [_SQUARE]))

    assert lines == ["LINESTRING EMPTY"]


def test_a_z_that_is_not_finite_only_counts_without_ignore_z() -> None:
    a = _line_strings("a", [[(0, 0, math.nan), (10, 0, 1)]], axes="xyz")
    b = _wkt("b", PointXYZ(), ["POINT Z (0 1 2)"])
    df = pl.DataFrame([a, b])

    with pytest.raises(ComputeError, match="expects finite coordinates"):
        df.select(geo.hausdorff_line_between("a", "b"))
    out = df.select(geo.hausdorff_line_between("a", "b", ignore_z=True))
    assert _as_wkt(out["a"]) == ["LINESTRING Z(10 0 1,0 1 2)"]
    out = df.select(geo.hausdorff_line_between("a", "b", non_finite="skip"))
    assert _as_wkt(out["a"]) == ["LINESTRING Z(10 0 1,0 1 2)"]


def test_a_failing_row_gives_null_with_ignore_per_row_errors() -> None:
    a = _line_strings("a", [[(0, 0), (math.nan, 1), (10, 0)], [(0, 0), (10, 0)]])
    b = _wkt("b", LineStringXY(), [_BUMP, _BUMP])

    out = pl.DataFrame([a, b]).select(
        pl.col("a").geo.hausdorff_line_between("b", ignore_per_row_errors=True)
    )

    assert _as_wkt(out["a"]) == [None, "LINESTRING(5 0,5 4)"]


def test_against_a_single_failing_geometry_every_row_is_null_with_ignore_per_row_errors() -> (
    None
):
    one = _line_strings("one", [[(0, 0), (math.nan, 1), (10, 0)]])
    df = pl.DataFrame(_wkt("many", PointXY(), ["POINT (3 0)", None]))

    out = df.select(geo.hausdorff_line_between(one, "many", ignore_per_row_errors=True))

    assert _as_wkt(out.to_series()) == [None, None]


def test_an_unknown_non_finite_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="`non_finite` must be one of"):
        geo.hausdorff_line_between("a", "b", non_finite="drop")  # type: ignore[arg-type]


def test_the_line_carries_the_crs() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (3 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:28992"), geo.set_crs("b", "EPSG:28992"))

    out = df.select(geo.hausdorff_line_between("a", "b"))

    assert out.schema["a"] == LineStringXY(crs="EPSG:28992")


def test_different_crss_are_refused() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"), geo.set_crs("b", "EPSG:3857"))

    with pytest.raises(ValueError, match="different CRSs"):
        df.select(geo.hausdorff_line_between("a", "b"))


def test_unequal_lengths_are_refused() -> None:
    a = _wkt("a", PointXY(), ["POINT (0 0)", "POINT (1 1)"])
    b = _wkt("b", PolygonXY(), [_SQUARE, _SQUARE, _SQUARE])

    with pytest.raises(ComputeError, match="cannot pair 2 geometries with 3"):
        pl.select(geo.hausdorff_line_between(a, b))


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
            _wkt(
                "a",
                _ORACLE_DTYPES[a_kind],
                list(shapely.to_wkt(a, rounding_precision=-1)),
            ),
            _wkt(
                "b",
                _ORACLE_DTYPES[b_kind],
                list(shapely.to_wkt(b, rounding_precision=-1)),
            ),
        ]
    )

    out = shapely.from_wkt(
        _as_wkt(df.select(geo.hausdorff_line_between("a", "b"))["a"])
    )

    either_empty = shapely.is_empty(a) | shapely.is_empty(b)
    assert all(shapely.is_empty(out) == either_empty)
    kept = ~either_empty
    out, a, b = out[kept], np.asarray(a)[kept], np.asarray(b)[kept]
    starts = shapely.get_point(out, 0)
    ends = shapely.get_point(out, 1)
    expected = shapely.hausdorff_distance(a, b)
    assert np.allclose(shapely.length(out), expected, rtol=0, atol=1e-12)
    assert np.all(shapely.distance(starts, a) < 1e-12)
    assert np.all(shapely.distance(ends, b) < 1e-12)
