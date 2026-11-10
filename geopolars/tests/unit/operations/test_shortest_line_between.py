"""The shortest line between two geometries, forwarded to GEOS."""

from __future__ import annotations

import math
import random
from typing import Any

import polars as pl
import pytest
import shapely
import shapely.affinity
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal

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
    return _as_wkt(df.select(geo.shortest_line_between("a", "b", **kwargs))["a"])


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

    out = pl.DataFrame([a, b]).select(geo.shortest_line_between("a", "b"))

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

    out = df.select(geo.shortest_line_between(one, "many"))
    assert _as_wkt(out.to_series()) == [
        "LINESTRING(1 0,3 0)",
        None,
        "LINESTRING(0 1,0 3)",
        "LINESTRING EMPTY",
    ]
    out = df.select(geo.shortest_line_between("many", one))
    assert _as_wkt(out["many"]) == [
        "LINESTRING(3 0,1 0)",
        None,
        "LINESTRING(0 3,0 1)",
        "LINESTRING EMPTY",
    ]


@pytest.mark.parametrize(
    "one",
    [
        _wkt("one", PolygonXY(), ["POLYGON EMPTY"]),
        _wkt("one", PointXY(), ["POINT EMPTY"]),
    ],
)
def test_a_single_empty_geometry_gives_empty_lines(one: pl.Series) -> None:
    df = pl.DataFrame(_wkt("many", PointXY(), ["POINT (3 0)", None]))

    out = df.select(geo.shortest_line_between(one, "many"))
    assert _as_wkt(out.to_series()) == ["LINESTRING EMPTY", None]
    out = df.select(geo.shortest_line_between("many", one))
    assert _as_wkt(out["many"]) == ["LINESTRING EMPTY", None]


def test_columns_chunked_differently_are_paired_row_by_row() -> None:
    a = _chunked([_wkt("a", PointXY(), ["POINT (3 0)"]), _wkt("a", PointXY(), [None])])
    b = _wkt("b", PolygonXY(), [_SQUARE, _SQUARE])

    out = pl.DataFrame([a, b]).select(geo.shortest_line_between("a", "b"))

    assert _as_wkt(out["a"]) == ["LINESTRING(3 0,1 0)", None]


# Only x is NaN, as NaN in both x and y is an empty point.
# Every distance to it is NaN, so GEOS finds no nearest points and fails.
_NOT_FINITE_POINT = pl.Series("a", [{"x": math.nan, "y": 1.0}]).ext.to(PointXY())


def _line_strings(name: str, rows: list[list[tuple[float, float]]]) -> pl.Series:
    storage = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
    coords = [[{"x": x, "y": y} for x, y in row] for row in rows]
    return pl.Series(name, coords, dtype=storage).ext.to(LineStringXY())


# GEOS fails to build a linestring of a single vertex,
# and the early finite check fails on the NaN vertex of the other.
_FAILING_LINES = [[(0, 0)], [(0, 0), (math.nan, 1), (2, 2)]]


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_against_a_single_null_geometry_no_row_reaches_geos(
    early_finite_check: bool,
) -> None:
    one = _wkt("one", PolygonXY(), [None])
    df = pl.DataFrame(_line_strings("many", _FAILING_LINES))
    kwargs = {"early_finite_check": early_finite_check}

    out = df.select(geo.shortest_line_between("many", one, **kwargs))
    assert out["many"].to_list() == [None, None]
    assert out.schema["many"] == LineStringXY()
    out = df.select(geo.shortest_line_between(one, "many", **kwargs))
    assert out.to_series().to_list() == [None, None]


def test_a_single_null_geometry_against_no_rows_gives_no_lines() -> None:
    df = pl.DataFrame(_wkt("many", PointXY(), []))
    one = _wkt("one", PolygonXY(), [None])

    assert df.select(geo.shortest_line_between("many", one)).height == 0
    assert df.select(geo.shortest_line_between(one, "many")).height == 0


def test_against_the_first_of_no_rows_gives_no_lines() -> None:
    df = pl.DataFrame(
        [_wkt("a", PointXY(), ["POINT (3 0)"]), _wkt("b", PolygonXY(), [_SQUARE])]
    )

    out = df.filter(False).select(
        "a",
        geo.shortest_line_between("a", pl.col("b").first()).alias("line"),
    )

    assert out.height == 0


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_a_row_paired_with_a_null_never_reaches_geos(early_finite_check: bool) -> None:
    df = pl.DataFrame(
        [
            _wkt("a", PointXY(), [None, None, "POINT (3 0)"]),
            _line_strings("b", [*_FAILING_LINES, [(0, 0), (1, 0), (1, 1)]]),
        ]
    )

    out = df.select(
        geo.shortest_line_between("a", "b", early_finite_check=early_finite_check)
    )

    assert _as_wkt(out["a"]) == [None, None, "LINESTRING(3 0,1 0)"]


@pytest.mark.parametrize(
    "one",
    [
        _wkt("one", PolygonXY(), ["POLYGON EMPTY"]),
        _wkt("one", PointXY(), ["POINT EMPTY"]),
    ],
)
@pytest.mark.parametrize("early_finite_check", [False, True])
def test_against_a_single_empty_geometry_no_row_reaches_geos(
    one: pl.Series, early_finite_check: bool
) -> None:
    df = pl.DataFrame(_line_strings("many", _FAILING_LINES))
    kwargs = {"early_finite_check": early_finite_check}

    out = df.select(geo.shortest_line_between("many", one, **kwargs))
    assert _as_wkt(out["many"]) == ["LINESTRING EMPTY", "LINESTRING EMPTY"]
    out = df.select(geo.shortest_line_between(one, "many", **kwargs))
    assert _as_wkt(out.to_series()) == ["LINESTRING EMPTY", "LINESTRING EMPTY"]


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_a_row_paired_with_an_empty_geometry_never_reaches_geos(
    early_finite_check: bool,
) -> None:
    df = pl.DataFrame(
        [
            _wkt("a", PointXY(), ["POINT EMPTY", "POINT EMPTY", "POINT (3 0)"]),
            _line_strings("b", [*_FAILING_LINES, [(0, 0), (1, 0), (1, 1)]]),
        ]
    )

    out = df.select(
        geo.shortest_line_between("a", "b", early_finite_check=early_finite_check)
    )

    assert _as_wkt(out["a"]) == [
        "LINESTRING EMPTY",
        "LINESTRING EMPTY",
        "LINESTRING(3 0,1 0)",
    ]


def _without_vertices(name: str, dtype: GeoArrowType, row: list[Any]) -> pl.Series:
    """Two rows of `row`, whose parts have no vertices, as `dtype`."""
    storage: pl.DataType = pl.Struct({"x": pl.Float64, "y": pl.Float64})
    for _ in range(dtype._nesting):
        storage = pl.List(storage)
    return pl.Series(name, [row, row], dtype=storage).ext.to(dtype)


@pytest.mark.parametrize(
    "a",
    [
        _without_vertices("a", PolygonXY(), [[]]),
        _without_vertices("a", MultiLineStringXY(), [[]]),
        _without_vertices("a", MultiPolygonXY(), [[]]),
        _without_vertices("a", MultiPolygonXY(), [[[]]]),
    ],
    ids=["polygon", "multilinestring", "multipolygon", "multipolygon_with_a_ring"],
)
@pytest.mark.parametrize("early_finite_check", [False, True])
def test_a_row_paired_with_parts_without_vertices_never_reaches_geos(
    a: pl.Series, early_finite_check: bool
) -> None:
    df = pl.DataFrame([a, _line_strings("b", _FAILING_LINES)])
    kwargs = {"early_finite_check": early_finite_check}

    out = df.select(geo.shortest_line_between("a", "b", **kwargs))

    assert _as_wkt(out["a"]) == ["LINESTRING EMPTY", "LINESTRING EMPTY"]


def test_an_empty_geometry_against_a_failing_one_is_empty_with_ignore_per_row_errors() -> (
    None
):
    df = pl.DataFrame(
        [
            _wkt("a", PolygonXY(), ["POLYGON EMPTY"]),
            _line_strings("b", _FAILING_LINES[:1]),
        ]
    )

    out = df.select(geo.shortest_line_between("a", "b", ignore_per_row_errors=True))

    assert _as_wkt(out["a"]) == ["LINESTRING EMPTY"]


def test_geos_raises_on_a_geometry_that_is_not_finite() -> None:
    b = _wkt("b", PolygonXY(), [_SQUARE])

    with pytest.raises(ComputeError, match="GEOSNearestPoints_r failed"):
        pl.select(geo.shortest_line_between(_NOT_FINITE_POINT, b))


def test_the_early_finite_check_raises_before_geos() -> None:
    b = _wkt("b", PolygonXY(), [_SQUARE])

    with pytest.raises(ComputeError, match="GEOS expects finite x and y"):
        pl.select(
            geo.shortest_line_between(_NOT_FINITE_POINT, b, early_finite_check=True)
        )


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_a_failing_row_gives_null_with_ignore_per_row_errors(
    early_finite_check: bool,
) -> None:
    a = pl.concat([_NOT_FINITE_POINT, _wkt("a", PointXY(), ["POINT (3 0)"])])
    df = pl.DataFrame([a, _wkt("b", PolygonXY(), [_SQUARE, _SQUARE])])

    out = df.select(
        pl.col("a").geo.shortest_line_between(
            "b",
            ignore_per_row_errors=True,
            early_finite_check=early_finite_check,
        )
    )

    assert _as_wkt(out["a"]) == [None, "LINESTRING(3 0,1 0)"]


@pytest.mark.parametrize("early_finite_check", [False, True])
def test_against_a_single_geometry_a_failing_row_gives_null_with_ignore_per_row_errors(
    early_finite_check: bool,
) -> None:
    many = pl.concat([_NOT_FINITE_POINT, _wkt("a", PointXY(), ["POINT (3 0)"])])
    df = pl.DataFrame(many.alias("many"))
    one = _wkt("one", PolygonXY(), [_SQUARE])
    kwargs = {"ignore_per_row_errors": True, "early_finite_check": early_finite_check}

    out = df.select(geo.shortest_line_between(one, "many", **kwargs))
    assert _as_wkt(out.to_series()) == [None, "LINESTRING(1 0,3 0)"]
    out = df.select(geo.shortest_line_between("many", one, **kwargs))
    assert _as_wkt(out["many"]) == [None, "LINESTRING(3 0,1 0)"]


def test_the_line_carries_the_crs() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (3 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:28992"), geo.set_crs("b", "EPSG:28992"))

    out = df.select(geo.shortest_line_between("a", "b"))

    assert out.schema["a"] == LineStringXY(crs="EPSG:28992")


def test_different_crss_are_refused() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"), geo.set_crs("b", "EPSG:3857"))

    with pytest.raises(ValueError, match="different CRSs"):
        df.select(geo.shortest_line_between("a", "b"))


def test_a_crs_on_only_one_side_is_refused() -> None:
    df = pl.DataFrame(
        [_wkt("a", PolygonXY(), [_SQUARE]), _wkt("b", PointXY(), ["POINT (0 0)"])]
    ).with_columns(geo.set_crs("a", "EPSG:4326"))

    with pytest.raises(ValueError, match="only one of the geometries"):
        df.select(geo.shortest_line_between("a", "b"))


def _with_crs(name: str, dtype: GeoArrowType, wkt: str, crs: str) -> pl.Series:
    return pl.select(geo.set_crs(_wkt(name, dtype, [wkt]), crs)).to_series()


# At 60°N, (0 61) is 111 km away and (1.5 60) 84 km,
# but in longitude/latitude (0 61) is the nearer one.
_AT_60_NORTH = pl.DataFrame(
    [
        _with_crs("a", PointXY(), "POINT (0 60)", "EPSG:4326"),
        _with_crs("b", MultiPointXY(), "MULTIPOINT ((0 61), (1.5 60))", "EPSG:4326"),
    ]
)


def test_the_line_is_found_and_given_in_in_crs() -> None:
    out = _AT_60_NORTH.select(geo.shortest_line_between("a", "b", in_crs="EPSG:3857"))

    expected = _with_crs("a", LineStringXY(), "LINESTRING (0 60, 1.5 60)", "EPSG:4326")
    expected = pl.select(geo.to_crs(expected, "EPSG:3857")).to_series()
    assert out.schema["a"] == LineStringXY(crs="EPSG:3857")
    assert_series_equal(out["a"], expected)


def test_by_default_the_line_is_found_in_the_coordinates() -> None:
    out = _AT_60_NORTH.select(geo.shortest_line_between("a", "b"))

    assert out.schema["a"] == LineStringXY(crs="EPSG:4326")
    assert _as_wkt(out["a"]) == ["LINESTRING(0 60,0 61)"]


def test_the_line_is_given_in_any_in_crs() -> None:
    out = _AT_60_NORTH.select(geo.shortest_line_between("a", "b", in_crs="EPSG:28992"))

    assert out.schema["a"] == LineStringXY(crs="EPSG:28992")


def test_different_crss_are_accepted_with_in_crs() -> None:
    df = pl.DataFrame(
        [
            _with_crs("a", PointXY(), "POINT (0 0)", "EPSG:4326"),
            _with_crs("b", PointXY(), "POINT (1000 0)", "EPSG:3857"),
        ]
    )

    out = df.select(geo.shortest_line_between("a", "b", in_crs="EPSG:3857"))

    assert _as_wkt(out["a"]) == ["LINESTRING(0 0,1000 0)"]


@pytest.mark.parametrize(("b_crs", "reprojected"), [("EPSG:3857", 0), ("EPSG:4326", 1)])
def test_a_geometry_already_in_in_crs_is_not_reprojected(
    b_crs: str, reprojected: int
) -> None:
    lf = pl.LazyFrame(
        [
            _with_crs("a", PointXY(), "POINT (0 0)", "EPSG:3857"),
            _with_crs("b", PointXY(), "POINT (1 0)", b_crs),
        ]
    )

    plan = lf.select(geo.shortest_line_between("a", "b", in_crs="EPSG:3857")).explain()

    assert plan.count(":to_crs(") == reprojected


@pytest.mark.parametrize("side", ["a", "b"])
def test_a_geometry_without_a_crs_is_refused_with_in_crs(side: str) -> None:
    df = pl.DataFrame(
        [
            _wkt("a", PointXY(), ["POINT (0 0)"]),
            _with_crs("b", PointXY(), "POINT (1 0)", "EPSG:4326"),
        ]
    )
    a, b = ("a", "b") if side == "a" else ("b", "a")

    with pytest.raises(ValueError, match="both geometries need a CRS"):
        df.select(geo.shortest_line_between(a, b, in_crs="EPSG:3857"))


def test_z_does_not_need_a_flag_to_be_reprojected() -> None:
    df = pl.DataFrame(
        [
            _with_crs("a", PointXYZ(), "POINT Z (0 60 100)", "EPSG:4979"),
            _with_crs("b", PointXY(), "POINT (1 60)", "EPSG:4326"),
        ]
    )

    out = df.select(geo.shortest_line_between("a", "b", in_crs="EPSG:3857"))

    assert out.schema["a"] == LineStringXY(crs="EPSG:3857")


def test_z_does_not_need_a_flag_to_be_reprojected_to_a_crs_with_z() -> None:
    df = pl.DataFrame(
        [
            _with_crs("a", PointXYZ(), "POINT Z (0 60 100)", "EPSG:4326"),
            _with_crs("b", PointXY(), "POINT (1 60)", "EPSG:4326"),
        ]
    )

    out = df.select(geo.shortest_line_between("a", "b", in_crs="EPSG:4979"))

    assert out.schema["a"] == LineStringXY(crs="EPSG:4979")


def test_nulls_and_empty_geometries_survive_the_reprojection() -> None:
    many = _wkt("many", PointXY(), ["POINT (3 60)", None, "POINT EMPTY"])
    many = pl.select(geo.set_crs(many, "EPSG:4326")).to_series()
    one = _with_crs(
        "one", PolygonXY(), "POLYGON ((0 59, 1 59, 1 61, 0 61, 0 59))", "EPSG:4326"
    )
    df = pl.DataFrame(many)

    out = df.select(geo.shortest_line_between("many", one, in_crs="EPSG:3857"))
    assert out.schema["many"] == LineStringXY(crs="EPSG:3857")
    lines = _as_wkt(out["many"])
    assert lines[0] is not None
    assert lines[1:] == [None, "LINESTRING EMPTY"]
    out = df.select(geo.shortest_line_between(one, "many", in_crs="EPSG:3857"))
    assert _as_wkt(out.to_series())[1:] == [None, "LINESTRING EMPTY"]


def test_unequal_lengths_are_refused() -> None:
    a = _wkt("a", PointXY(), ["POINT (0 0)", "POINT (1 1)"])
    b = _wkt("b", PolygonXY(), [_SQUARE, _SQUARE, _SQUARE])

    with pytest.raises(ComputeError, match="cannot pair 2 geometries with 3"):
        pl.select(geo.shortest_line_between(a, b))


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

    out = shapely.from_wkt(_as_wkt(df.select(geo.shortest_line_between("a", "b"))["a"]))

    expected = shapely.shortest_line(a, b)
    expected = [shapely.LineString() if e is None else e for e in expected]
    assert all(shapely.equals_exact(out, expected, tolerance=1e-12))
