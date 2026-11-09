"""The buffer of a geometry: every point within a distance of it."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.exceptions import ComputeError, ShapeError

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PolygonXY,
)
from tests.unit.conftest import Dimension

if TYPE_CHECKING:
    from polars._typing import EngineType

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


def _buffered(df: pl.DataFrame, distance: float, **kwargs: object) -> list:
    out = df.select(geo.buffer("geom", distance, **kwargs))  # type: ignore[arg-type]
    return out.to_series().ext.storage().to_list()


def _areas(df: pl.DataFrame, distance: float, **kwargs: object) -> list[float]:
    out = df.select(geo.buffer("geom", distance, **kwargs).geo.area_planar())  # type: ignore[arg-type]
    return out.to_series().to_list()


_ORIGIN = {"x": 0.0, "y": 0.0}
_EMPTY_POINT = {"x": math.nan, "y": math.nan}
_SQUARE = _ring((0, 0), (4, 0), (4, 4), (0, 4))
_SEGMENT = _line((0, 0), (4, 0))
# A ring that touches itself at (2, 0): two triangles meeting at a point.
_TOUCHING_ITSELF = _ring((0, 0), (4, 0), (4, 4), (2, 0), (0, 4))
# Two 4 by 4 squares, joined by a neck 0.4 wide.
_DUMBBELL = _ring(
    (0, 0), (4, 0), (4, 1.8), (6, 1.8), (6, 0), (10, 0),
    (10, 4), (6, 4), (6, 2.2), (4, 2.2), (4, 4), (0, 4),
)  # fmt: skip


def _inscribed_area(quadrant_segments: int) -> float:
    """The area of the regular polygon GEOS draws for a unit circle."""
    n = 4 * quadrant_segments
    return n / 2 * math.sin(2 * math.pi / n)


def test_the_output_is_xy_polygons_or_multipolygons(
    dimension: Dimension,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> None:
    for df, expected in (
        (coords.select(dimension.point()), PolygonXY()),
        (dimension.lines(line_coords), PolygonXY()),
        (dimension.polygons(ring_coords), PolygonXY()),
        (dimension.multipoints(line_coords), MultiPolygonXY()),
        (dimension.multilinestrings(ring_coords), MultiPolygonXY()),
        (dimension.multipolygons(multipolygon_coords), MultiPolygonXY()),
    ):
        out = df.select(geo.buffer(df.columns[0], 1.0, ignore_per_row_errors=True))

        assert out.schema[df.columns[0]] == expected


@pytest.mark.parametrize(
    ("distance", "single"),
    [(-1.0, MultiPolygonXY()), (0.0, MultiPolygonXY()), (1.0, PolygonXY())],
)
def test_allow_splitting_gives_multipolygons_for_a_negative_distance(
    distance: float,
    single: GeoArrowType,
    dimension: Dimension,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> None:
    for df, expected in (
        (coords.select(dimension.point()), single),
        (dimension.lines(line_coords), single),
        (dimension.polygons(ring_coords), single),
        (dimension.multipoints(line_coords), MultiPolygonXY()),
        (dimension.multilinestrings(ring_coords), MultiPolygonXY()),
        (dimension.multipolygons(multipolygon_coords), MultiPolygonXY()),
    ):
        buffered = geo.buffer(
            df.columns[0], distance, allow_splitting=True, ignore_per_row_errors=True
        )

        assert df.select(buffered).schema[df.columns[0]] == expected


def test_the_crs_is_carried_over() -> None:
    df = _column(PolygonXY, [_SQUARE], crs="EPSG:28992")

    out = df.select(geo.buffer("geom", 1.0))

    assert out.schema["geom"] == PolygonXY(crs="EPSG:28992")


@pytest.mark.parametrize("quadrant_segments", [1, 2, 8])
def test_a_point_gets_a_regular_polygon(quadrant_segments: int) -> None:
    df = _column(PointXY, _ORIGIN)

    [[ring]] = _buffered(df, 1.0, quadrant_segments=quadrant_segments)
    [area] = _areas(df, 1.0, quadrant_segments=quadrant_segments)

    assert len(ring) == 4 * quadrant_segments + 1
    assert area == pytest.approx(_inscribed_area(quadrant_segments))


@pytest.mark.parametrize("distance", [0.0, -1.0])
@pytest.mark.parametrize(
    ("dtype", "geometry"), [(PointXY, _ORIGIN), (LineStringXY, _SEGMENT)]
)
def test_a_point_or_line_gets_an_empty_polygon_without_a_positive_distance(
    distance: float, dtype: type[GeoArrowType], geometry: object
) -> None:
    assert _buffered(_column(dtype, geometry), distance) == [[]]


@pytest.mark.parametrize(
    ("dtype", "empty"),
    [
        (PointXY, _EMPTY_POINT),
        (LineStringXY, []),
        (PolygonXY, []),
        (MultiPointXY, []),
        (MultiLineStringXY, []),
        (MultiPolygonXY, []),
    ],
)
def test_a_null_or_empty_geometry_stays_so(
    dtype: type[GeoArrowType], empty: object
) -> None:
    assert _buffered(_column(dtype, None, empty), 1.0) == [None, []]


def test_a_negative_distance_shrinks_a_polygon() -> None:
    [[ring]] = _buffered(_column(PolygonXY, [_SQUARE]), -1.0)

    assert sorted((p["x"], p["y"]) for p in ring[:-1]) == [
        (1, 1), (1, 3), (3, 1), (3, 3)
    ]  # fmt: skip


def test_a_polygon_split_in_two_raises() -> None:
    df = _column(PolygonXY, [_DUMBBELL], [_SQUARE])

    with pytest.raises(ComputeError, match="split into 2 parts"):
        _buffered(df, -0.5)


def test_a_polygon_split_in_two_is_null_when_ignoring_errors() -> None:
    df = _column(PolygonXY, [_DUMBBELL], [_SQUARE])

    [split, square] = _buffered(df, -0.5, ignore_per_row_errors=True)

    assert split is None
    assert len(square) == 1


def test_a_multipolygon_split_in_two_gets_both_parts() -> None:
    [parts] = _buffered(_column(MultiPolygonXY, [[_DUMBBELL]]), -0.5)

    assert len(parts) == 2


def test_a_polygon_split_in_two_gets_both_parts_with_allow_splitting() -> None:
    df = _column(PolygonXY, [_DUMBBELL], [_SQUARE])

    [split, square] = _buffered(df, -0.5, allow_splitting=True)

    assert len(split) == 2
    assert len(square) == 1


def test_a_point_shrunk_with_allow_splitting_is_an_empty_multipolygon() -> None:
    assert _buffered(_column(PointXY, _ORIGIN), -1.0, allow_splitting=True) == [[]]


def test_multipoints_that_overlap_merge() -> None:
    df = _column(MultiPointXY, _line((0, 0), (1, 0)), _line((0, 0), (10, 0)))

    [close, apart] = _buffered(df, 1.0)

    assert len(close) == 1
    assert len(apart) == 2


@pytest.mark.parametrize(
    ("cap_style", "area"), [("flat", 8.0), ("square", 12.0), (None, 8.0 + math.pi)]
)
def test_the_cap_style_shapes_the_ends_of_a_line(
    cap_style: str | None, area: float
) -> None:
    [out] = _areas(_column(LineStringXY, _SEGMENT), 1.0, cap_style=cap_style)

    assert out == pytest.approx(area, rel=0.01)


@pytest.mark.parametrize(
    ("join_style", "area"),
    [("mitre", 36.0), ("bevel", 34.0), ("round", 32.0 + math.pi)],
)
def test_the_join_style_shapes_the_corners(join_style: str, area: float) -> None:
    [out] = _areas(_column(PolygonXY, [_SQUARE]), 1.0, join_style=join_style)

    assert out == pytest.approx(area, rel=0.01)


def test_a_low_mitre_limit_bevels_the_corners() -> None:
    df = _column(PolygonXY, [_SQUARE])

    [area] = _areas(df, 1.0, join_style="mitre", mitre_limit=1.0)

    assert area < 36.0


@pytest.mark.parametrize(("side", "y"), [("left", 1.0), ("right", -1.0)])
def test_one_side_of_a_line(side: str, y: float) -> None:
    df = _column(LineStringXY, _SEGMENT)

    [[ring]] = _buffered(df, 1.0, side=side)
    [area] = _areas(df, 1.0, side=side)

    assert {p["y"] for p in ring} == {0.0, y}
    assert area == pytest.approx(4.0)


def test_one_side_of_a_point_is_its_regular_buffer() -> None:
    df = _column(PointXY, _ORIGIN)

    assert _buffered(df, 1.0, side="left") == _buffered(df, 1.0)


def test_the_exterior_is_clockwise_unless_ccw_outer() -> None:
    df = _column(PointXY, _ORIGIN)

    for ccw_outer in (False, True):
        out = df.select(geo.buffer("geom", 1.0, ccw_outer=ccw_outer).geo.is_ccw())

        assert out.to_series().to_list() == [ccw_outer]


def test_ccw_outer_turns_every_part_of_a_multipolygon() -> None:
    df = _column(MultiPolygonXY, [[_DUMBBELL]])

    out = df.select(geo.buffer("geom", -0.5, ccw_outer=True).geo.is_ccw())

    assert out.to_series().to_list() == [True]


def test_a_non_finite_vertex_is_caught_early() -> None:
    df = _column(LineStringXY, _line((0, 0), (1, math.nan)), _SEGMENT)

    with pytest.raises(ComputeError, match="finite"):
        _buffered(df, 1.0, early_finite_check=True)

    [caught, line] = _buffered(
        df, 1.0, early_finite_check=True, ignore_per_row_errors=True
    )
    assert caught is None
    assert line is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance": math.nan},
        {"distance": math.inf},
        {"quadrant_segments": 0},
        {"cap_style": "butt"},
        {"join_style": "miter"},
        {"side": "up"},
        {"mitre_limit": 2.0},
        {"mitre_limit": 2.0, "join_style": "bevel"},
        {"side": "left", "cap_style": "flat"},
        {"side": "right", "distance": -1.0},
    ],
)
def test_invalid_options_are_refused(kwargs: dict[str, object]) -> None:
    distance = kwargs.pop("distance", 1.0)

    with pytest.raises(ValueError, match="buffer"):
        geo.buffer("geom", distance, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dtype", "geometry"),
    [
        (PolygonXY, [_SQUARE]),
        (MultiPolygonXY, [[_SQUARE]]),
        (MultiPointXY, _line((0, 0))),
    ],
)
def test_one_side_is_refused_without_lines(
    dtype: type[GeoArrowType], geometry: object
) -> None:
    with pytest.raises(TypeError, match="one side"):
        _buffered(_column(dtype, geometry), 1.0, side="left")


def test_a_ring_touching_itself_splits_at_zero_with_allow_splitting() -> None:
    df = _column(PolygonXY, [_TOUCHING_ITSELF])

    [parts] = _buffered(df, 0.0, allow_splitting=True)

    assert len(parts) == 2


def _points_with_distances(*distances: float | None) -> pl.DataFrame:
    points = _column(PointXY, *[_ORIGIN] * len(distances))
    return points.with_columns(pl.Series("d", distances, dtype=pl.Float64))


def test_a_distance_column_gives_every_row_its_own_distance() -> None:
    df = _points_with_distances(1.0, 2.0, None)

    out = df.select(geo.buffer("geom", "d").geo.area_planar())

    [one, two, null] = out.to_series().to_list()
    assert one == pytest.approx(_inscribed_area(8))
    assert two == pytest.approx(4 * _inscribed_area(8))
    assert null is None


def test_a_distance_series_of_one_value_is_broadcast() -> None:
    df = _column(PointXY, _ORIGIN, _ORIGIN)

    out = df.select(geo.buffer("geom", pl.Series([2.0])).geo.area_planar())

    assert out.to_series().to_list() == pytest.approx([4 * _inscribed_area(8)] * 2)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_a_distance_column_of_another_length_raises(engine: EngineType) -> None:
    lf = _column(PointXY, _ORIGIN, _ORIGIN, _ORIGIN).lazy()

    with pytest.raises(ShapeError):
        lf.select(geo.buffer("geom", pl.Series([1.0, 2.0]))).collect(engine=engine)


def test_an_integer_distance_works() -> None:
    df = _column(PointXY, _ORIGIN).with_columns(d=pl.lit(1))

    assert (
        _buffered(df, 1)
        == df.select(geo.buffer("geom", "d")).to_series().ext.storage().to_list()
    )


def test_a_distance_column_with_allow_splitting_always_gives_multipolygons() -> None:
    df = _points_with_distances(1.0, 2.0)

    out = df.select(geo.buffer("geom", "d", allow_splitting=True))

    assert out.schema["geom"] == MultiPolygonXY()
    assert [len(parts) for parts in out.to_series().ext.storage().to_list()] == [1, 1]


def test_a_distance_column_splits_a_polygon_with_allow_splitting() -> None:
    df = _column(PolygonXY, [_DUMBBELL], [_SQUARE]).with_columns(
        d=pl.Series([-0.5, 1.0])
    )

    out = df.select(geo.buffer("geom", "d", allow_splitting=True))

    assert [len(parts) for parts in out.to_series().ext.storage().to_list()] == [2, 1]


@pytest.mark.parametrize(
    ("distance", "kwargs", "match"),
    [
        (math.nan, {}, "finite"),
        (math.inf, {}, "finite"),
        (-1.0, {"side": "left"}, "at least 0"),
    ],
)
def test_an_invalid_row_distance_raises_or_is_null(
    distance: float, kwargs: dict[str, object], match: str
) -> None:
    df = _column(LineStringXY, _SEGMENT, _SEGMENT).with_columns(
        d=pl.Series([distance, 1.0])
    )

    with pytest.raises(ComputeError, match=match):
        df.select(geo.buffer("geom", "d", **kwargs))  # type: ignore[arg-type]

    out = df.select(geo.buffer("geom", "d", ignore_per_row_errors=True, **kwargs))  # type: ignore[arg-type]
    [invalid, valid] = out.to_series().ext.storage().to_list()
    assert invalid is None
    assert valid is not None


@pytest.mark.parametrize("side", ["left", "right"])
def test_one_side_by_column_matches_one_side_by_constant(side: str) -> None:
    df = _column(LineStringXY, _SEGMENT).with_columns(d=pl.lit(1.0))

    by_column = df.select(geo.buffer("geom", "d", side=side))  # type: ignore[arg-type]

    assert by_column.to_series().ext.storage().to_list() == _buffered(
        df, 1.0, side=side
    )
