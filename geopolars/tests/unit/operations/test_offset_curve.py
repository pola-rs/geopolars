"""The offset curve of a line: the line running alongside it at a distance."""

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


def _column(
    dtype: type[GeoArrowType], *rows: object, crs: str | None = None
) -> pl.DataFrame:
    """A `geom` column of `dtype`, one row per storage value."""
    storage = pl.Series("geom", list(rows), dtype=dtype._geo_storage)
    return pl.DataFrame(storage.ext.to(dtype(crs=crs)))


def _offset(df: pl.DataFrame, distance: object, **kwargs: object) -> list:
    out = df.select(geo.offset_curve("geom", distance, **kwargs))  # type: ignore[arg-type]
    return out.to_series().ext.storage().to_list()


_SEGMENT = _line((0, 0), (4, 0))
_CORNER = _line((0, 0), (4, 0), (4, 4))
# Crosses itself, so its offset to the right comes in two pieces.
_CROSSING = _line((0, 0), (4, 4), (4, 0), (0, 4))


def test_the_output_is_xy_linestrings_or_multilinestrings(
    dimension: Dimension, line_coords: pl.DataFrame, ring_coords: pl.DataFrame
) -> None:
    for df, allow_splitting, expected in (
        (dimension.lines(line_coords), False, LineStringXY()),
        (dimension.lines(line_coords), True, MultiLineStringXY()),
        (dimension.multilinestrings(ring_coords), False, MultiLineStringXY()),
        (dimension.multilinestrings(ring_coords), True, MultiLineStringXY()),
    ):
        curve = geo.offset_curve(
            df.columns[0],
            1.0,
            allow_splitting=allow_splitting,
            ignore_per_row_errors=True,
        )

        assert df.select(curve).schema[df.columns[0]] == expected


def test_the_crs_is_carried_over() -> None:
    df = _column(LineStringXY, _SEGMENT, crs="EPSG:28992")

    out = df.select(geo.offset_curve("geom", 1.0))

    assert out.schema["geom"] == LineStringXY(crs="EPSG:28992")


@pytest.mark.parametrize(("distance", "y"), [(1.0, 1.0), (-1.0, -1.0), (0.0, 0.0)])
def test_a_positive_distance_offsets_to_the_left(distance: float, y: float) -> None:
    [curve] = _offset(_column(LineStringXY, _SEGMENT), distance)

    assert curve == _line((0, y), (4, y))


@pytest.mark.parametrize(
    ("join_style", "expected"),
    [
        ("mitre", _line((0, -1), (5, -1), (5, 4))),
        ("bevel", _line((0, -1), (4, -1), (5, 0), (5, 4))),
    ],
)
def test_the_join_style_shapes_the_corners(join_style: str, expected: Line) -> None:
    [curve] = _offset(_column(LineStringXY, _CORNER), -1.0, join_style=join_style)

    assert curve == expected


@pytest.mark.parametrize("quadrant_segments", [8, 16])
def test_a_round_corner_has_quadrant_segments(quadrant_segments: int) -> None:
    [curve] = _offset(
        _column(LineStringXY, _CORNER), -1.0, quadrant_segments=quadrant_segments
    )

    # The two straight ends, plus the quarter circle between them.
    assert len(curve) == 3 + quadrant_segments


def test_the_inside_of_a_corner_is_cut_short() -> None:
    [curve] = _offset(_column(LineStringXY, _CORNER), 1.0)

    assert curve == _line((0, 1), (3, 1), (3, 4))


def test_a_split_line_raises() -> None:
    with pytest.raises(ComputeError, match="split into 2 parts"):
        _offset(_column(LineStringXY, _CROSSING), -1.0)


def test_a_split_line_is_null_when_ignoring_errors() -> None:
    df = _column(LineStringXY, _CROSSING, _SEGMENT)

    [split, segment] = _offset(df, -1.0, ignore_per_row_errors=True)

    assert split is None
    assert segment is not None


def test_a_split_line_gets_both_parts_with_allow_splitting() -> None:
    [curve] = _offset(_column(LineStringXY, _CROSSING), -1.0, allow_splitting=True)

    assert len(curve) == 2


def test_every_part_of_a_multilinestring_gets_its_curve() -> None:
    df = _column(MultiLineStringXY, [_SEGMENT, _CROSSING])

    [curve] = _offset(df, -1.0)

    assert curve[0] == _line((0, -1), (4, -1))
    assert len(curve) == 3


@pytest.mark.parametrize(
    ("dtype", "empty"), [(LineStringXY, []), (MultiLineStringXY, [])]
)
def test_a_null_or_empty_geometry_stays_so(
    dtype: type[GeoArrowType], empty: object
) -> None:
    assert _offset(_column(dtype, None, empty), 1.0) == [None, []]


def test_a_line_without_length_gives_an_empty_linestring() -> None:
    assert _offset(_column(LineStringXY, _line((1, 1), (1, 1))), 1.0) == [[]]


def test_a_distance_column_gives_every_row_its_own_distance() -> None:
    df = _column(LineStringXY, _SEGMENT, _SEGMENT, _SEGMENT).with_columns(
        d=pl.Series([1.0, -2.0, None])
    )

    [left, right, null] = _offset(df, "d")

    assert left == _line((0, 1), (4, 1))
    assert right == _line((0, -2), (4, -2))
    assert null is None


def test_a_non_finite_row_distance_raises_or_is_null() -> None:
    df = _column(LineStringXY, _SEGMENT, _SEGMENT).with_columns(
        d=pl.Series([math.nan, 1.0])
    )

    with pytest.raises(ComputeError, match="finite"):
        _offset(df, "d")

    [invalid, valid] = _offset(df, "d", ignore_per_row_errors=True)
    assert invalid is None
    assert valid is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance": math.nan},
        {"distance": math.inf},
        {"quadrant_segments": 7},
        {"join_style": "miter"},
        {"mitre_limit": 2.0},
        {"mitre_limit": 2.0, "join_style": "bevel"},
    ],
)
def test_invalid_options_are_refused(kwargs: dict[str, object]) -> None:
    distance = kwargs.pop("distance", 1.0)

    with pytest.raises(ValueError, match="offset_curve"):
        geo.offset_curve("geom", distance, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dtype", "geometry"),
    [
        (PointXY, {"x": 0.0, "y": 0.0}),
        (MultiPointXY, _line((0, 0))),
        (PolygonXY, [_line((0, 0), (1, 0), (0, 1), (0, 0))]),
    ],
)
def test_anything_but_lines_is_refused(
    dtype: type[GeoArrowType], geometry: object
) -> None:
    with pytest.raises(TypeError, match="offset_curve expects a linestring"):
        _offset(_column(dtype, geometry), 1.0)
