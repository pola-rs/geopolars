"""The point at a distance along a line."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.exceptions import ComputeError

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYZM,
    MultiLineStringXY,
    MultiPointXY,
    PointXY,
    PolygonXY,
)

if TYPE_CHECKING:
    from tests.unit.conftest import Dimension

Line = list[dict[str, float]]


def _line(*xy: tuple[float, float]) -> Line:
    """An open linestring through `xy`."""
    return [{"x": x, "y": y} for x, y in xy]


def _point(x: float, y: float) -> dict[str, float]:
    return {"x": x, "y": y}


def _column(
    dtype: type[GeoArrowType], *rows: object, crs: str | None = None
) -> pl.DataFrame:
    """A `geom` column of `dtype`, one row per storage value."""
    storage = pl.Series("geom", list(rows), dtype=dtype._geo_storage)
    return pl.DataFrame(storage.ext.to(dtype(crs=crs)))


def _interpolate(df: pl.DataFrame, distance: object, **kwargs: object) -> list:
    out = df.select(geo.interpolate("geom", distance, **kwargs))  # type: ignore[arg-type]
    return out.to_series().ext.storage().to_list()


def _is_empty(point: dict[str, float]) -> bool:
    return math.isnan(point["x"]) and math.isnan(point["y"])


_SEGMENT = _line((0, 0), (4, 0))
# 8 long: 4 along x, then 4 along y.
_CORNER = _line((0, 0), (4, 0), (4, 4))


def test_the_output_is_a_point_of_the_input_dimension(
    dimension: Dimension, line_coords: pl.DataFrame, ring_coords: pl.DataFrame
) -> None:
    for df in (
        dimension.lines(line_coords),
        dimension.multilinestrings(ring_coords),
    ):
        point = geo.interpolate(df.columns[0], 1.0)

        assert df.select(point).schema[df.columns[0]] == dimension.point_dtype()


def test_the_crs_is_carried_over() -> None:
    df = _column(LineStringXY, _SEGMENT, crs="EPSG:28992")

    out = df.select(geo.interpolate("geom", 1.0))

    assert out.schema["geom"] == PointXY(crs="EPSG:28992")


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, _point(0, 0)),
        (2.0, _point(2, 0)),
        (6.0, _point(4, 2)),
        (-1.0, _point(4, 3)),
        (100.0, _point(4, 4)),
        (-100.0, _point(0, 0)),
    ],
)
def test_an_absolute_distance_is_measured_along_the_line(
    distance: float, expected: dict[str, float]
) -> None:
    assert _interpolate(_column(LineStringXY, _CORNER), distance) == [expected]


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.0, _point(0, 0)),
        (0.25, _point(2, 0)),
        (0.75, _point(4, 2)),
        (1.0, _point(4, 4)),
        (-0.25, _point(4, 2)),
        (2.0, _point(4, 4)),
    ],
)
def test_a_normalized_distance_is_a_fraction_of_the_length(
    fraction: float, expected: dict[str, float]
) -> None:
    df = _column(LineStringXY, _CORNER)

    assert _interpolate(df, fraction, mode="normalized") == [expected]


def test_z_and_m_are_interpolated() -> None:
    line = [
        {"x": 0.0, "y": 0.0, "z": 10.0, "m": 0.0},
        {"x": 4.0, "y": 0.0, "z": 14.0, "m": 8.0},
    ]

    [point] = _interpolate(_column(LineStringXYZM, line), 1.0)

    assert point == {"x": 1.0, "y": 0.0, "z": 11.0, "m": 2.0}


@pytest.mark.parametrize(
    ("distance", "mode"), [(6.0, "absolute"), (0.75, "normalized")]
)
def test_a_multilinestring_is_walked_part_after_part(
    distance: float, mode: str
) -> None:
    df = _column(MultiLineStringXY, [_SEGMENT, _line((10, 0), (10, 4))])

    assert _interpolate(df, distance, mode=mode) == [_point(10, 2)]


@pytest.mark.parametrize("mode", ["absolute", "normalized"])
def test_a_line_without_length_gives_its_vertex(mode: str) -> None:
    df = _column(LineStringXY, _line((1, 1), (1, 1)))

    assert _interpolate(df, 0.5, mode=mode) == [_point(1, 1)]


@pytest.mark.parametrize("dtype", [LineStringXY, MultiLineStringXY])
def test_a_null_or_empty_geometry_gives_a_null_or_empty_point(
    dtype: type[GeoArrowType],
) -> None:
    [null, empty] = _interpolate(_column(dtype, None, []), 1.0)

    assert null is None
    assert _is_empty(empty)


def test_a_multilinestring_of_empty_parts_gives_an_empty_point() -> None:
    [point] = _interpolate(_column(MultiLineStringXY, [[]]), 1.0)

    assert _is_empty(point)



@pytest.mark.parametrize(
    ("parts", "distance", "expected"),
    [
        ([[], _SEGMENT], 1.0, _point(1, 0)),
        ([[], _SEGMENT], 0.0, _point(0, 0)),
        ([_SEGMENT, []], 100.0, _point(4, 0)),
        ([_SEGMENT, [], _line((10, 0), (10, 4))], 6.0, _point(10, 2)),
    ],
)
def test_empty_parts_are_skipped(
    parts: list[Line], distance: float, expected: dict[str, float]
) -> None:
    df = _column(MultiLineStringXY, parts)

    assert _interpolate(df, distance) == [expected]


def test_a_distance_column_gives_every_row_its_own_distance() -> None:
    df = _column(LineStringXY, _SEGMENT, _SEGMENT, _SEGMENT).with_columns(
        d=pl.Series([1.0, -1.0, None])
    )

    assert _interpolate(df, "d") == [_point(1, 0), _point(3, 0), None]


def test_a_non_finite_row_distance_raises_or_is_null() -> None:
    df = _column(LineStringXY, _SEGMENT, _SEGMENT).with_columns(
        d=pl.Series([math.nan, 1.0])
    )

    with pytest.raises(ComputeError, match="finite"):
        _interpolate(df, "d")

    assert _interpolate(df, "d", ignore_per_row_errors=True) == [None, _point(1, 0)]


def test_the_namespace_forwards() -> None:
    df = _column(LineStringXY, _CORNER)

    out = df.select(pl.col("geom").geo.interpolate(0.5, mode="normalized"))

    assert out.to_series().ext.storage().to_list() == [_point(4, 0)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance": math.nan},
        {"distance": math.inf},
        {"mode": "fraction"},
    ],
)
def test_invalid_options_are_refused(kwargs: dict[str, object]) -> None:
    distance = kwargs.pop("distance", 1.0)

    with pytest.raises(ValueError, match="interpolate"):
        geo.interpolate("geom", distance, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dtype", "geometry"),
    [
        (PointXY, _point(0, 0)),
        (MultiPointXY, _line((0, 0))),
        (PolygonXY, [_line((0, 0), (1, 0), (0, 1), (0, 0))]),
    ],
)
def test_anything_but_lines_is_refused(
    dtype: type[GeoArrowType], geometry: object
) -> None:
    with pytest.raises(TypeError, match="interpolate expects a linestring"):
        _interpolate(_column(dtype, geometry), 1.0)
