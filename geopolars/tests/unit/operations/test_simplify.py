"""The simplification of a geometry: fewer vertices, within a tolerance."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.exceptions import ComputeError

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYZ,
    MultiLineStringXY,
    MultiLineStringXYZ,
    MultiPolygonXY,
    MultiPolygonXYZ,
    PolygonXY,
    PolygonXYZ,
    PolygonXYZM,
)
from tests.unit.conftest import Dimension

Ring = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Ring:
    """A closed ring through `xy`."""
    return [{"x": x, "y": y} for x, y in (*xy, xy[0])]


def _line(*xy: tuple[float, float]) -> Ring:
    """An open linestring through `xy`."""
    return [{"x": x, "y": y} for x, y in xy]


def _column(
    dtype: type[GeoArrowType], *rows: object, crs: str | None = None
) -> pl.DataFrame:
    """A `geom` column of `dtype`, one row per storage value."""
    storage = pl.Series("geom", list(rows), dtype=dtype._geo_storage)
    return pl.DataFrame(storage.ext.to(dtype(crs=crs)))


def _simplified(df: pl.DataFrame, tolerance: float, **kwargs: bool) -> list:
    out = df.select(geo.simplify("geom", tolerance, **kwargs))
    return out.to_series().ext.storage().to_list()


# Every vertex but the ends is within 0.01 of the line between them.
_ZIGZAG = _line((0, 0), (1, 0.01), (2, 0), (3, 0.01), (4, 0))
_STRAIGHT = _line((0, 0), (4, 0))
# A dent of 0.1 in the top edge.
_DENTED = _ring((0, 0), (4, 0), (4, 4), (2, 3.9), (0, 4))
_SQUARE = _ring((0, 0), (4, 0), (4, 4), (0, 4))
_TINY = _ring((0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1))
_BIG = _ring((0, 0), (10, 0), (10, 10), (0, 10))
_SMALL_HOLE = _ring((5, 5), (5.1, 5), (5.1, 5.1), (5, 5.1))
# The bottom edge dips 0.9, a notch from the top reaches below it to -0.5.
# Straightening the bottom edge cuts the notch through, into two polygons.
_CUT_BY_NOTCH = _ring(
    (0, 0), (5, -0.9), (10, 0), (10, 5), (5.2, 5), (5, -0.5), (4.8, 5), (0, 5)
)


@pytest.mark.parametrize("preserve_topology", [False, True])
def test_a_linestring_keeps_only_its_endpoints(preserve_topology: bool) -> None:
    df = _column(LineStringXY, _ZIGZAG)

    simplified = _simplified(df, 0.5, preserve_topology=preserve_topology)

    assert simplified == [_STRAIGHT]


def test_a_tolerance_of_zero_keeps_every_vertex() -> None:
    assert _simplified(_column(LineStringXY, _ZIGZAG), 0.0) == [_ZIGZAG]


@pytest.mark.parametrize("preserve_topology", [False, True])
@pytest.mark.parametrize(
    ("dtype", "geometry", "simple"),
    [
        (LineStringXY, _ZIGZAG, _STRAIGHT),
        (MultiLineStringXY, [_ZIGZAG], [_STRAIGHT]),
        (PolygonXY, [_DENTED], [_SQUARE]),
        (MultiPolygonXY, [[_DENTED]], [[_SQUARE]]),
    ],
)
def test_a_null_or_empty_geometry_stays_so(
    preserve_topology: bool,
    dtype: type[GeoArrowType],
    geometry: list,
    simple: list,
) -> None:
    df = _column(dtype, None, [], geometry)

    simplified = _simplified(df, 0.5, preserve_topology=preserve_topology)

    assert simplified == [None, [], simple]


def test_a_dent_is_straightened() -> None:
    assert _simplified(_column(PolygonXY, [_DENTED]), 0.5) == [[_SQUARE]]


def test_a_tiny_polygon_becomes_empty() -> None:
    assert _simplified(_column(PolygonXY, [_TINY], [_DENTED]), 0.5) == [[], [_SQUARE]]


def test_a_tiny_polygon_is_kept_with_preserve_topology() -> None:
    [[ring]] = _simplified(_column(PolygonXY, [_TINY]), 0.5, preserve_topology=True)

    assert len(ring) >= 4


def test_a_tiny_hole_is_dropped() -> None:
    df = _column(PolygonXY, [_BIG, _SMALL_HOLE])

    assert _simplified(df, 0.5) == [[_BIG]]


def test_a_tiny_hole_is_kept_with_preserve_topology() -> None:
    df = _column(PolygonXY, [_BIG, _SMALL_HOLE])

    [rings] = _simplified(df, 0.5, preserve_topology=True)

    assert rings[0] == _BIG
    assert len(rings) == 2


def test_a_polygon_cut_in_two_raises() -> None:
    df = _column(PolygonXY, [_CUT_BY_NOTCH], [_DENTED])

    with pytest.raises(ComputeError, match="MultiPolygon"):
        _simplified(df, 1.0)


def test_a_polygon_cut_in_two_is_null_when_ignoring_errors() -> None:
    df = _column(PolygonXY, [_CUT_BY_NOTCH], [_DENTED])

    simplified = _simplified(df, 1.0, on_row_error="null")

    assert simplified == [None, [_SQUARE]]


def test_a_polygon_is_not_cut_with_preserve_topology() -> None:
    df = _column(PolygonXY, [_CUT_BY_NOTCH])

    assert _simplified(df, 1.0, preserve_topology=True) == [[_CUT_BY_NOTCH]]


def test_a_multipolygon_cut_in_two_gets_both_parts() -> None:
    [parts] = _simplified(_column(MultiPolygonXY, [[_CUT_BY_NOTCH]]), 1.0)

    assert len(parts) == 2


def test_a_tiny_part_of_a_multipolygon_is_dropped() -> None:
    df = _column(MultiPolygonXY, [[_TINY], [_DENTED]])

    assert _simplified(df, 0.5) == [[[_SQUARE]]]


def test_a_tiny_part_of_a_multipolygon_is_kept_with_preserve_topology() -> None:
    df = _column(MultiPolygonXY, [[_TINY], [_DENTED]])

    [parts] = _simplified(df, 0.5, preserve_topology=True)

    assert len(parts) == 2
    assert parts[1] == [_SQUARE]


def test_a_multilinestring_is_simplified_part_by_part() -> None:
    other = _line((0, 4), (2, 4.01), (4, 4))
    df = _column(MultiLineStringXY, [_ZIGZAG, other])

    assert _simplified(df, 0.5) == [[_STRAIGHT, _line((0, 4), (4, 4))]]


def test_z_is_kept_but_m_is_not() -> None:
    xy = [(0, 0), (4, 0), (4, 4), (2, 3.9), (0, 4), (0, 0)]
    ring = [
        {"x": x, "y": y, "z": 10.0 * x, "m": float(i)} for i, (x, y) in enumerate(xy)
    ]
    df = _column(PolygonXYZM, [ring])

    out = df.select(geo.simplify("geom", 0.5))

    assert out.schema["geom"] == PolygonXYZ()
    assert out.to_series().ext.storage().to_list() == [
        [
            [
                {"x": 0, "y": 0, "z": 0},
                {"x": 4, "y": 0, "z": 40},
                {"x": 4, "y": 4, "z": 40},
                {"x": 0, "y": 4, "z": 0},
                {"x": 0, "y": 0, "z": 0},
            ]
        ]
    ]


def test_the_output_has_no_m(
    dimension: Dimension,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> None:
    z = dimension.has_z
    for df, expected in (
        (dimension.lines(line_coords), LineStringXYZ() if z else LineStringXY()),
        (
            dimension.multilinestrings(ring_coords),
            MultiLineStringXYZ() if z else MultiLineStringXY(),
        ),
        (dimension.polygons(ring_coords), PolygonXYZ() if z else PolygonXY()),
        (
            dimension.multipolygons(multipolygon_coords),
            MultiPolygonXYZ() if z else MultiPolygonXY(),
        ),
    ):
        out = df.select(geo.simplify(df.columns[0], 0.5))

        assert out.schema[df.columns[0]] == expected


def test_points_and_multipoints_are_returned_as_they_are(
    dimension: Dimension, coords: pl.DataFrame, line_coords: pl.DataFrame
) -> None:
    for df in (coords.select(dimension.point()), dimension.multipoints(line_coords)):
        out = df.select(geo.simplify(df.columns[0], 100.0))

        assert out.equals(df)


def test_the_crs_is_carried_over() -> None:
    df = _column(PolygonXY, [_DENTED], crs="EPSG:28992")

    out = df.select(geo.simplify("geom", 0.5))

    assert out.schema["geom"] == PolygonXY(crs="EPSG:28992")


def test_a_geometry_geos_fails_on_raises() -> None:
    # GEOS cannot build a linestring of one vertex.
    df = _column(LineStringXY, _line((1, 1)), _ZIGZAG)

    with pytest.raises(ComputeError, match="point array"):
        _simplified(df, 0.5)


def test_a_geometry_geos_fails_on_is_null_when_ignoring_errors() -> None:
    df = _column(LineStringXY, _line((1, 1)), _ZIGZAG)

    assert _simplified(df, 0.5, on_row_error="null") == [None, _STRAIGHT]


@pytest.mark.parametrize("tolerance", [-1.0, math.nan])
def test_rejects_a_negative_or_nan_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance of at least 0"):
        geo.simplify("geom", tolerance)


def test_the_namespace_forwards_preserve_topology() -> None:
    df = _column(PolygonXY, [_CUT_BY_NOTCH])

    assert df.select(pl.col("geom").geo.simplify(1.0, preserve_topology=True)).equals(
        df.select(geo.simplify("geom", 1.0, preserve_topology=True))
    )


def test_the_namespace_forwards_on_row_error() -> None:
    df = _column(PolygonXY, [_CUT_BY_NOTCH])

    with pytest.raises(ComputeError, match="MultiPolygon"):
        df.select(pl.col("geom").geo.simplify(1.0))
    assert df.select(pl.col("geom").geo.simplify(1.0, on_row_error="null")).equals(
        df.select(geo.simplify("geom", 1.0, on_row_error="null"))
    )
