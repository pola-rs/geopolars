"""The convex hull of a linestring or polygon, or their multi-variants."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError

from geopolars import geo
from geopolars.datatypes import (
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PolygonXY,
    PolygonXYZ,
    PolygonXYZM,
)
from tests.unit.conftest import Dimension

_XY_RINGS = pl.List(pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64})))

Ring = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Ring:
    """A closed ring through `xy`."""
    return [{"x": x, "y": y} for x, y in (*xy, xy[0])]


# Concave: a notch at (2, 2).
_NOTCHED = _ring((0, 0), (4, 0), (4, 4), (2, 2), (0, 4))
_HULL = _ring((0, 0), (0, 4), (4, 4), (4, 0))
_ALONG_A_LINE = _ring((0, 0), (1, 0), (2, 0))


def _polygons(*polygons: list[Ring] | None, crs: str | None = None) -> pl.DataFrame:
    """A polygon column, one row per list of XY rings."""
    return pl.DataFrame({"rings": list(polygons)}, schema={"rings": _XY_RINGS}).select(
        geo.polygons_from_rings_in_single_column("rings", crs=crs).alias("polygon")
    )


def _multipolygons(
    *multis: list[list[Ring]] | None, crs: str | None = None
) -> pl.DataFrame:
    """A `polygon` column of `MultiPolygonXY`, one row per list of polygons."""
    storage = pl.Series("polygon", list(multis), dtype=pl.List(_XY_RINGS))
    return pl.DataFrame(storage.ext.to(MultiPolygonXY(crs=crs)))


def _line(*xy: tuple[float, float]) -> Ring:
    """An open linestring through `xy`."""
    return [{"x": x, "y": y} for x, y in xy]


def _lines(*lines: Ring | None) -> pl.DataFrame:
    """A `polygon` column of `LineStringXY`, one row per list of vertices.
    Named like the polygon columns, so `_hulls` reads it too."""
    storage = pl.Series("polygon", list(lines), dtype=LineStringXY._geo_storage)
    return pl.DataFrame(storage.ext.to(LineStringXY()))


def _multipoints(*multis: Ring | None) -> pl.DataFrame:
    """A `polygon` column of `MultiPointXY`, one row per list of points."""
    storage = pl.Series("polygon", list(multis), dtype=MultiPointXY._geo_storage)
    return pl.DataFrame(storage.ext.to(MultiPointXY()))


def _multilines(*multis: list[Ring] | None) -> pl.DataFrame:
    """A `polygon` column of `MultiLineStringXY`, one row per list of linestrings."""
    storage = pl.Series("polygon", list(multis), dtype=MultiLineStringXY._geo_storage)
    return pl.DataFrame(storage.ext.to(MultiLineStringXY()))


def _hulls(df: pl.DataFrame) -> list:
    return df.select(geo.convex_hull("polygon")).to_series().ext.storage().to_list()


def test_a_concave_polygon_loses_its_notch() -> None:
    assert _hulls(_polygons([_NOTCHED])) == [[_HULL]]


def test_a_null_or_empty_polygon() -> None:
    assert _hulls(_polygons(None, [], [_NOTCHED])) == [None, [], [_HULL]]


def test_a_polygon_that_encloses_nothing_gives_a_null_hull() -> None:
    assert _hulls(_polygons([_ALONG_A_LINE], [_NOTCHED])) == [None, [_HULL]]


def test_z_is_kept_but_m_is_not() -> None:
    xy = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4), (0, 0)]
    ring = [
        {"x": x, "y": y, "z": 10.0 * x, "m": float(i)} for i, (x, y) in enumerate(xy)
    ]
    storage = pl.Series("polygon", [[ring]], dtype=PolygonXYZM._geo_storage)
    df = pl.DataFrame(storage.ext.to(PolygonXYZM()))

    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXYZ()
    assert out.to_series().ext.storage().to_list() == [
        [
            [
                {"x": 0, "y": 0, "z": 0},
                {"x": 0, "y": 4, "z": 0},
                {"x": 4, "y": 4, "z": 40},
                {"x": 4, "y": 0, "z": 40},
                {"x": 0, "y": 0, "z": 0},
            ]
        ]
    ]


def test_a_hull_has_no_m(
    dimension: Dimension,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> None:
    expected = PolygonXYZ() if dimension.has_z else PolygonXY()

    for df in (
        dimension.lines(line_coords),
        dimension.multipoints(line_coords),
        dimension.multilinestrings(ring_coords),
        dimension.polygons(ring_coords),
        dimension.multipolygons(multipolygon_coords),
    ):
        out = df.select(geo.convex_hull(df.columns[0]))

        assert out.schema[df.columns[0]] == expected


def test_the_crs_is_carried_over() -> None:
    out = _polygons([_NOTCHED], crs="EPSG:28992").select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY(crs="EPSG:28992")


def test_rejects_a_point_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"x": [1.0], "y": [2.0]}).select(
        geo.point("x", "y").alias("point")
    )

    with pytest.raises(
        TypeError, match="convex_hull expects a linestring, polygon, multipoint"
    ):
        lf.select(geo.convex_hull("point")).collect_schema()


def test_a_multipoint_is_hulled_around_its_outer_points() -> None:
    # The corners of a square, and one point inside it.
    points = _line((0, 0), (4, 0), (2, 2), (4, 4), (0, 4))

    df = _multipoints(points)
    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY()
    assert _hulls(df) == [[_HULL]]


def test_a_null_or_empty_multipoint() -> None:
    square = _line((0, 0), (4, 0), (4, 4), (0, 4))

    assert _hulls(_multipoints(None, [], square)) == [None, [], [_HULL]]


def test_one_or_two_points_give_a_null_hull() -> None:
    square = _line((0, 0), (4, 0), (4, 4), (0, 4))

    hulls = _hulls(_multipoints(_line((1, 1)), _line((0, 0), (1, 1)), square))

    assert hulls == [None, None, [_HULL]]


def test_a_bent_linestring_is_closed_around() -> None:
    bent = _line((0, 0), (4, 0), (4, 4), (0, 4))

    df = _lines(bent)
    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY()
    assert _hulls(df) == [[_HULL]]


def test_a_null_or_empty_linestring() -> None:
    bent = _line((0, 0), (4, 0), (4, 4), (0, 4))

    assert _hulls(_lines(None, [], bent)) == [None, [], [_HULL]]


def test_a_straight_linestring_gives_a_null_hull() -> None:
    straight = _line((0, 0), (1, 0), (2, 0))
    single = _line((1, 1))
    bent = _line((0, 0), (4, 0), (4, 4), (0, 4))

    assert _hulls(_lines(straight, single, bent)) == [None, None, [_HULL]]


def test_a_multilinestring_gets_one_hull_of_the_whole() -> None:
    bottom = _line((0, 0), (4, 0))
    top = _line((0, 4), (4, 4))

    df = _multilines([bottom, top], None, [], [_line((0, 0), (1, 0))])
    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY()
    assert _hulls(df) == [[_HULL], None, [], None]


def test_a_multipolygon_gets_one_hull_of_the_whole() -> None:
    left = [_ring((0, 0), (1, 0), (1, 1), (0, 1))]
    right = [_ring((3, 0), (4, 0), (4, 1), (3, 1))]

    df = _multipolygons([left, right])
    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY()
    assert _hulls(df) == [[_ring((0, 0), (0, 1), (4, 1), (4, 0))]]


def test_a_null_or_empty_multipolygon() -> None:
    assert _hulls(_multipolygons(None, [], [[_NOTCHED]])) == [None, [], [_HULL]]


def test_a_multipolygon_that_encloses_nothing_gives_a_null_hull() -> None:
    flat = [[_ALONG_A_LINE], [_ring((3, 0), (4, 0), (5, 0))]]

    assert _hulls(_multipolygons(flat, [[_NOTCHED]])) == [None, [_HULL]]


def test_a_multipolygon_hull_carries_the_crs() -> None:
    df = _multipolygons([[_NOTCHED]], crs="EPSG:28992")

    out = df.select(geo.convex_hull("polygon"))

    assert out.schema["polygon"] == PolygonXY(crs="EPSG:28992")


# GEOS fails on a vertex with just one NaN axis.
_HALF_NAN = _line((0, 0), (4, 0), (2, float("nan")), (4, 4), (0, 4))


_WITH_AN_EMPTY_POINT = _line((0, 0), (4, 0), (float("nan"), float("nan")), (0, 4))


def test_finite_leaves_an_empty_point_to_geos() -> None:
    df = _multipoints(_WITH_AN_EMPTY_POINT)

    out = df.select(geo.convex_hull("polygon", early_coord_check="finite"))

    triangle = _line((0, 0), (0, 4), (4, 0), (0, 0))
    assert out.to_series().ext.storage().to_list() == [[triangle]]


def test_finite_and_empty_refuses_an_empty_point() -> None:
    df = _multipoints(_WITH_AN_EMPTY_POINT)

    with pytest.raises(ComputeError, match="GEOS expects finite x and y"):
        df.select(geo.convex_hull("polygon", early_coord_check="finite_and_empty"))


def test_finite_still_refuses_a_vertex_with_one_nan_axis() -> None:
    with pytest.raises(ComputeError, match="GEOS expects finite x and y"):
        _lines(_HALF_NAN).select(geo.convex_hull("polygon", early_coord_check="finite"))


def test_a_geometry_geos_fails_on_raises_by_default() -> None:
    df = _lines(_HALF_NAN, _line((0, 0), (4, 0), (4, 4), (0, 4)))

    with pytest.raises(ComputeError, match="NaN"):
        df.select(geo.convex_hull("polygon"))


def test_on_row_error_null_gives_a_null_hull() -> None:
    df = _lines(_HALF_NAN, _line((0, 0), (4, 0), (4, 4), (0, 4)))

    hulls = (
        df.select(geo.convex_hull("polygon", on_row_error="null"))
        .to_series()
        .ext.storage()
        .to_list()
    )

    assert hulls == [None, [_HULL]]


def test_the_namespace_matches_the_function() -> None:
    df = _polygons([_NOTCHED], None)

    assert df.select(pl.col("polygon").geo.convex_hull()).equals(
        df.select(geo.convex_hull("polygon"))
    )


def _is_ccw(df: pl.DataFrame) -> list[bool | None]:
    return df.select(geo.is_ccw("polygon")).to_series().to_list()


def test_the_exterior_runs_as_geos_gives_it_clockwise() -> None:
    out = _polygons([_NOTCHED]).select(geo.convex_hull("polygon"))

    assert _is_ccw(out) == [False]


def test_ccw_runs_the_exterior_counter_clockwise() -> None:
    out = _polygons([_NOTCHED]).select(
        geo.convex_hull("polygon", new_ring_orientation="ccw")
    )

    assert _is_ccw(out) == [True]
    assert out.to_series().ext.storage().to_list() == [
        [_ring((0, 0), (4, 0), (4, 4), (0, 4))]
    ]


def test_the_namespace_forwards_new_ring_orientation() -> None:
    df = _polygons([_NOTCHED])

    assert df.select(
        pl.col("polygon").geo.convex_hull(new_ring_orientation="ccw")
    ).equals(df.select(geo.convex_hull("polygon", new_ring_orientation="ccw")))


def test_the_namespace_forwards_on_row_error() -> None:
    df = _lines(_HALF_NAN)

    assert df.select(pl.col("polygon").geo.convex_hull(on_row_error="null")).equals(
        df.select(geo.convex_hull("polygon", on_row_error="null"))
    )
