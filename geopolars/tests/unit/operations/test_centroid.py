"""The centroid of a geometry: the centre of its length or area."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    MultiPolygonXYZ,
    PointXY,
    PolygonXYZ,
    PolygonXYZM,
)
from geopolars.geo.centroid import CentroidAnchor, CentroidMethod
from tests.unit.conftest import XY, Dimension, coordinates

if TYPE_CHECKING:
    from collections.abc import Callable

    from geopolars._typing import OnRowError

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)

Ring = list[dict[str, float]]

# How the errors for a geometry without a centroid start, one per reason.
_NO_LENGTH = "needs a geometry with length, got one without"
_NO_AREA = "needs a geometry with area, got one without"
_UNDERFLOW = "too small for a float"
_HOLES_EXCEED = "holes take out more area than its exterior holds"


def _ring(*xy: tuple[float, float]) -> Ring:
    """A closed ring through `xy`."""
    return [{"x": x, "y": y} for x, y in (*xy, xy[0])]


_SQUARE = _ring((0, 0), (1, 0), (1, 1), (0, 1))
# An L of three unit squares: its vertices average elsewhere than its area.
_L = _ring((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))
_BIG_SQUARE = _ring((10, 10), (12, 10), (12, 12), (10, 12))


def _polygons(*polygons: list[Ring] | None, crs: str | None = None) -> pl.DataFrame:
    """A polygon column, one row per list of XY rings."""
    return pl.DataFrame({"rings": list(polygons)}, schema={"rings": _XY_RINGS}).select(
        geo.polygons_from_rings_in_single_column("rings", crs=crs).alias("polygon")
    )


def _centroids(
    df: pl.DataFrame, on_row_error: OnRowError = "raise"
) -> list[tuple[float, float] | None]:
    out = (
        df.select(geo.centroid("polygon", on_row_error=on_row_error))
        .to_series()
        .ext.storage()
        .to_list()
    )
    return [None if c is None else (c["x"], c["y"]) for c in out]


@pytest.mark.parametrize(
    ("ring", "expected"),
    [
        (_SQUARE, (0.5, 0.5)),
        (_ring((0, 0), (3, 0), (0, 3)), (1.0, 1.0)),
        (_L, (5 / 6, 5 / 6)),
    ],
    ids=["square", "triangle", "L"],
)
def test_the_centroid_is_the_centre_of_the_area(
    ring: Ring, expected: tuple[float, float]
) -> None:
    (got,) = _centroids(_polygons([ring]))

    assert got == pytest.approx(expected)


def test_winding_order_does_not_change_the_centroid() -> None:
    assert _centroids(_polygons([_L[::-1]])) == _centroids(_polygons([_L]))


@pytest.mark.parametrize("reverse_hole", [False, True])
def test_a_hole_is_taken_out_whichever_way_it_winds(reverse_hole: bool) -> None:
    exterior = _ring((0, 0), (4, 0), (4, 4), (0, 4))
    hole = _ring((2, 2), (3, 2), (3, 3), (2, 3))
    # 16 at (2, 2), less 1 at (2.5, 2.5)
    expected = (16 * 2 - 2.5) / 15

    (got,) = _centroids(_polygons([exterior, hole[::-1] if reverse_hole else hole]))

    assert got == pytest.approx((expected, expected))


def test_a_polygon_keeps_its_dimension(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords)
    out = df.select(geo.centroid("polygon"))

    assert out.schema["polygon"] == dimension.point_dtype()
    # z and m don't move the centroid in xy.
    assert _centroids(df) == [
        pytest.approx(c) for c in _centroids(XY.polygons(ring_coords))
    ]


def _polygon_xyzm(
    rings: list[Ring],
    z: Callable[[float, float], float],
    m: Callable[[float, float], float] | None = None,
) -> pl.DataFrame:
    """A `PolygonXYZM` column of one polygon: `rings` with z at `z(x, y)`,
    and m at `m(x, y)`, or at `z(x, y)` as well without `m`."""
    m = m or z
    rings_zm = [
        [{**c, "z": z(c["x"], c["y"]), "m": m(c["x"], c["y"])} for c in ring]
        for ring in rings
    ]
    vertex = pl.Struct(dict.fromkeys("xyzm", pl.Float64))
    storage = pl.Series("polygon", [rings_zm], dtype=pl.List(pl.List(vertex)))
    return pl.DataFrame(storage.ext.to(PolygonXYZM()))


@pytest.mark.parametrize(
    ("rings", "z", "expected"),
    [
        # Over the L's centroid (5/6, 5/6); its vertices average to 1 + 2.
        ([_L], lambda x, y: x + 2 * y, 5 / 6 + 2 * 5 / 6),
        # Over the square with a hole's centroid; with the hole kept, it would be 2.
        (
            [
                _ring((0, 0), (4, 0), (4, 4), (0, 4)),
                _ring((2, 2), (3, 2), (3, 3), (2, 3)),
            ],
            lambda x, _: x,
            (16 * 2 - 2.5) / 15,
        ),
    ],
    ids=["L", "hole"],
)
@pytest.mark.parametrize("anchor", ["length_weighted_mean", "vertex_mean"])
@pytest.mark.parametrize("method", ["footprint", "3d"])
def test_z_and_m_are_averaged_over_the_area(
    rings: list[Ring],
    z: Callable[[float, float], float],
    expected: float,
    anchor: CentroidAnchor,
    method: CentroidMethod,
) -> None:
    # On a tilted plane, the centroid lies on it: above the centroid of the footprint.
    # As z and m change linearly over the polygon, the anchor makes no difference.
    out = _polygon_xyzm(rings, z).select(
        geo.centroid("polygon", method=method, anchor=anchor)
    )

    (got,) = out.to_series().ext.storage().to_list()

    assert (got["z"], got["m"]) == pytest.approx((expected, expected))


def _starting_at(ring: Ring, start: int) -> Ring:
    vertices = ring[:-1]
    rotated = vertices[start:] + vertices[:start]
    return [*rotated, rotated[0]]


def _saddle_z_and_m(rings: list[Ring], anchor: CentroidAnchor) -> tuple[float, float]:
    # x * y is not flat, but it is straight along the axis-aligned edges of the L.
    out = _polygon_xyzm(rings, lambda x, y: x * y).select(
        geo.centroid("polygon", anchor=anchor)
    )
    (got,) = out.to_series().ext.storage().to_list()
    return got["z"], got["m"]


@pytest.mark.parametrize("start", range(6))
@pytest.mark.parametrize("anchor", ["length_weighted_mean", "vertex_mean"])
def test_a_non_flat_polygon_does_not_depend_on_where_its_ring_starts(
    start: int, anchor: CentroidAnchor
) -> None:
    got = _saddle_z_and_m([_starting_at(_L, start)], anchor)

    assert got == pytest.approx((0.5, 0.5))


# The L, with a vertex added halfway along its edge from (2, 0) to (2, 1).
_L_SPLIT = _ring((0, 0), (2, 0), (2, 0.5), (2, 1), (1, 1), (1, 2), (0, 2))


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [("length_weighted_mean", 1 / 2), ("vertex_mean", 41 / 84)],
)
def test_only_the_vertex_mean_moves_when_an_edge_is_split(
    anchor: CentroidAnchor, expected: float
) -> None:
    assert _saddle_z_and_m([_L], anchor) == pytest.approx((0.5, 0.5))
    assert _saddle_z_and_m([_L_SPLIT], anchor) == pytest.approx((expected, expected))


_UTM = (500_000.0, 5_000_000.0)


def _square_at(origin: tuple[float, float], side: float) -> Ring:
    x, y = origin
    return _ring((x, y), (x + side, y), (x + side, y + side), (x, y + side))


@pytest.mark.parametrize(
    ("origin", "side"),
    [((1e9, 1e9), 1.0), (_UTM, 0.01)],
    ids=["unit-square-at-1e9", "centimetre-square-in-utm"],
)
def test_a_small_polygon_far_from_the_origin_keeps_its_centroid(
    origin: tuple[float, float], side: float
) -> None:
    expected = (origin[0] + side / 2, origin[1] + side / 2)

    (got,) = _centroids(_polygons([_square_at(origin, side)]))

    assert got == pytest.approx(expected, rel=0, abs=1e-6)


@pytest.mark.parametrize("method", ["footprint", "3d"])
def test_a_hole_that_nearly_fills_its_exterior_far_from_the_origin_keeps_its_centroid(
    method: CentroidMethod,
) -> None:
    # Powers of two keep every corner exact.
    corner, border = 2.0**17, 2.0**-20
    exterior = _square_at((corner, corner), 1.0)
    hole = _square_at((corner + border, corner + border), 1.0 - 2 * border)
    out = _polygon_xyzm([exterior, hole], lambda _x, _y: 100.0).select(
        geo.centroid("polygon", method=method)
    )

    (got,) = out.to_series().ext.storage().to_list()

    expected = (corner + 0.5, corner + 0.5, 100.0, 100.0)
    assert (got["x"], got["y"], got["z"], got["m"]) == pytest.approx(
        expected, rel=0, abs=1e-9
    )


@pytest.mark.parametrize("method", ["footprint", "3d"])
def test_a_flat_polygon_far_from_the_origin_keeps_its_z_and_m(
    method: CentroidMethod,
) -> None:
    corner = (_UTM[0] + 0.1, _UTM[1] + 0.7)
    out = _polygon_xyzm([_square_at(corner, 0.3)], lambda _x, _y: 100.0).select(
        geo.centroid("polygon", method=method)
    )

    (got,) = out.to_series().ext.storage().to_list()

    expected = (corner[0] + 0.15, corner[1] + 0.15, 100.0, 100.0)
    assert (got["x"], got["y"], got["z"], got["m"]) == pytest.approx(
        expected, rel=0, abs=1e-6
    )


# Polygons in 3D: measured on their own plane

XyzRing = list[tuple[float, float, float]]


def _xyz_ring(*xyz: tuple[float, float, float]) -> XyzRing:
    """A closed ring through `xyz`."""
    return [*xyz, xyz[0]]


def _xyz_vertices(ring: XyzRing) -> list[dict[str, float]]:
    return [dict(zip("xyz", c, strict=True)) for c in ring]


_XYZ_RINGS = pl.List(pl.List(pl.Struct(dict.fromkeys("xyz", pl.Float64))))


def _polygons_xyz(*polygons: list[XyzRing]) -> pl.DataFrame:
    """A `polygon` column of `PolygonXYZ`, one row per list of rings."""
    rows = [[_xyz_vertices(ring) for ring in polygon] for polygon in polygons]
    storage = pl.Series("polygon", rows, dtype=_XYZ_RINGS)
    return pl.DataFrame(storage.ext.to(PolygonXYZ()))


def _multipolygons_xyz(*multis: list[list[XyzRing]]) -> pl.DataFrame:
    """A `multi` column of `MultiPolygonXYZ`, one row per list of polygons."""
    rows = [
        [[_xyz_vertices(ring) for ring in polygon] for polygon in multi]
        for multi in multis
    ]
    storage = pl.Series("multi", rows, dtype=pl.List(_XYZ_RINGS))
    return pl.DataFrame(storage.ext.to(MultiPolygonXYZ()))


def _centroids_of_only_column(
    df: pl.DataFrame,
    method: CentroidMethod,
    anchor: CentroidAnchor = "length_weighted_mean",
    on_row_error: OnRowError = "raise",
) -> list[tuple[float, ...] | None]:
    (name,) = df.columns
    out = df.select(
        geo.centroid(name, method=method, anchor=anchor, on_row_error=on_row_error)
    )
    return [
        None if c is None else tuple(c.values())
        for c in out.to_series().ext.storage().to_list()
    ]


def test_a_vertical_polygon_only_has_a_centroid_in_3d() -> None:
    df = _polygons_xyz([_xyz_ring((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))])

    assert _centroids_of_only_column(df, "3d") == [pytest.approx((0.5, 0.0, 0.5))]
    with pytest.raises(pl.exceptions.ComputeError, match='area with `method="3d"`'):
        _centroids_of_only_column(df, "footprint")


@pytest.mark.parametrize("method", ["footprint", "3d"])
def test_a_small_polygon_keeps_its_centroid(method: CentroidMethod) -> None:
    # Small enough that the squared normal of the 3D fan underflows.
    side = 1e-90
    square = _xyz_ring((0, 0, 0), (side, 0, side), (side, side, side), (0, side, 0))
    df = _polygons_xyz([square])

    got = _centroids_of_only_column(df, method)

    assert got == [pytest.approx((side / 2,) * 3, rel=1e-12, abs=0)]


@pytest.mark.parametrize(
    ("rings", "error"),
    [
        ([_xyz_ring((0, 0, 0), (1e-200, 0, 0), (0, 0, 1e-200))], _UNDERFLOW),
        (
            [
                _xyz_ring((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
                _xyz_ring((-1, 0, -1), (2, 0, -1), (2, 0, 2), (-1, 0, 2)),
            ],
            _HOLES_EXCEED,
        ),
    ],
    ids=["tiny", "hole exceeds exterior"],
)
def test_a_polygon_without_area_has_no_centroid_in_3d(
    rings: list[XyzRing], error: str
) -> None:
    df = _polygons_xyz(rings)

    with pytest.raises(pl.exceptions.ComputeError, match=error):
        _centroids_of_only_column(df, "3d")
    assert _centroids_of_only_column(df, "3d", on_row_error="null") == [None]


@pytest.mark.parametrize(
    "exterior",
    [[], _xyz_ring((0, 0, 0), (2, 0, 0), (1, 0, 0))],
    ids=["empty", "collinear"],
)
@pytest.mark.parametrize("method", ["footprint", "3d"])
def test_a_hole_in_an_exterior_without_area_has_no_centroid(
    exterior: XyzRing, method: CentroidMethod
) -> None:
    hole = _xyz_ring((5, 5, 0), (6, 5, 0), (6, 6, 0), (5, 6, 0))
    df = _polygons_xyz([exterior, hole])

    with pytest.raises(pl.exceptions.ComputeError, match=_HOLES_EXCEED):
        _centroids_of_only_column(df, method)


@pytest.mark.parametrize("reverse_hole", [False, True])
def test_a_hole_in_a_vertical_polygon_is_taken_out_whichever_way_it_winds(
    reverse_hole: bool,
) -> None:
    exterior = _xyz_ring((0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4))
    hole = _xyz_ring((2, 0, 2), (3, 0, 2), (3, 0, 3), (2, 0, 3))
    # 16 at (2, 0, 2), less 1 at (2.5, 0, 2.5)
    expected = (16 * 2 - 2.5) / 15
    df = _polygons_xyz([exterior, hole[::-1] if reverse_hole else hole])

    assert _centroids_of_only_column(df, "3d") == [
        pytest.approx((expected, 0.0, expected))
    ]


def _saddle_centroid_in_3d(ring: Ring, anchor: CentroidAnchor) -> tuple[float, ...]:
    df = _polygon_xyzm([ring], lambda x, y: x * y)
    (got,) = _centroids_of_only_column(df, "3d", anchor)
    assert got is not None
    return got


@pytest.mark.parametrize("start", range(1, 6))
@pytest.mark.parametrize("anchor", ["length_weighted_mean", "vertex_mean"])
def test_a_non_flat_polygon_in_3d_does_not_depend_on_where_its_ring_starts(
    start: int, anchor: CentroidAnchor
) -> None:
    # Its plane is tilted, so the anchor moves every axis, not only z and m.
    expected = _saddle_centroid_in_3d(_L, anchor)

    got = _saddle_centroid_in_3d(_starting_at(_L, start), anchor)

    assert got == pytest.approx(expected)


def test_splitting_an_edge_does_not_move_the_length_weighted_3d_centroid() -> None:
    expected = _saddle_centroid_in_3d(_L, "length_weighted_mean")

    got = _saddle_centroid_in_3d(_L_SPLIT, "length_weighted_mean")

    assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    ("method", "expected"),
    # 1 at (0.5, 0.5, 0), and in 3D also 4 at (11, 0, 1)
    [("footprint", (0.5, 0.5, 0.0)), ("3d", (8.9, 0.1, 0.8))],
)
def test_a_vertical_part_of_a_multipolygon_only_weighs_in_3d(
    method: CentroidMethod, expected: tuple[float, float, float]
) -> None:
    flat = _xyz_ring((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
    upright = _xyz_ring((10, 0, 0), (12, 0, 0), (12, 0, 2), (10, 0, 2))
    df = _multipolygons_xyz([[flat], [upright]])

    assert _centroids_of_only_column(df, method) == [pytest.approx(expected)]


def test_the_crs_is_carried_over() -> None:
    df = _polygons([_SQUARE], crs="EPSG:4326")

    out = df.select(geo.centroid("polygon"))

    assert out.schema["polygon"] == PointXY._with_metadata_of(df.schema["polygon"])


def test_an_empty_polygon_has_the_empty_point_as_its_centroid() -> None:
    null, empty, empty_exterior, square = _centroids(
        _polygons(None, [], [[]], [_SQUARE])
    )

    assert null is None
    assert _all_nan(empty)
    assert _all_nan(empty_exterior)
    assert square == (0.5, 0.5)


def test_a_polygon_that_encloses_nothing_has_no_centroid() -> None:
    df = _polygons([_ring((0, 0), (2, 0), (1, 0))], [_SQUARE])

    with pytest.raises(pl.exceptions.ComputeError, match=_NO_AREA):
        _centroids(df)
    assert _centroids(df, on_row_error="null") == [None, (0.5, 0.5)]


# The cross products of a ring this small underflow to 0.
_TINY_TRIANGLE = _ring((0, 0), (1e-200, 0), (0, 1e-200))


# The area of the second one is still there, but the sums it is weighted by underflow.
@pytest.mark.parametrize("side", [1e-200, 1e-105])
def test_a_polygon_too_small_for_its_area_has_no_centroid(side: float) -> None:
    df = _polygons([_ring((0, 0), (side, 0), (0, side))], [_SQUARE])

    with pytest.raises(pl.exceptions.ComputeError, match=_UNDERFLOW):
        _centroids(df)
    assert _centroids(df, on_row_error="null") == [None, (0.5, 0.5)]


def test_a_polygon_whose_hole_exceeds_its_exterior_has_no_centroid() -> None:
    df = _polygons([_SQUARE, _BIG_SQUARE], [_SQUARE])

    with pytest.raises(pl.exceptions.ComputeError, match=_HOLES_EXCEED):
        _centroids(df)
    assert _centroids(df, on_row_error="null") == [None, (0.5, 0.5)]


def test_rows_without_a_centroid_line_up_across_chunks_and_slices() -> None:
    flat = [_ring((0, 0), (2, 0), (1, 0))]
    df = pl.concat(
        [_polygons([_SQUARE], flat, [_L]), _polygons([], flat, [_SQUARE])],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    _, nothing, l_shape, empty, nothing_again, square = _centroids(
        df, on_row_error="null"
    )

    assert (nothing, nothing_again) == (None, None)
    assert l_shape == pytest.approx((5 / 6, 5 / 6))
    assert _all_nan(empty)
    assert square == pytest.approx((0.5, 0.5))


def test_rows_without_a_centroid_line_up_across_many_rows() -> None:
    # Rows without a centroid are looked at again a block of rows at a time.
    def polygon(i: int) -> list[Ring]:
        if i % 11 == 0:
            return []
        if i % 7 == 0:
            return [_ring((i, 0), (i + 2, 0), (i + 1, 0))]
        return [_ring((i, 0), (i + 1, 0), (i + 1, 1), (i, 1))]

    rows = range(5000)

    centroids = _centroids(_polygons(*(polygon(i) for i in rows)), on_row_error="null")

    for i, centroid in zip(rows, centroids, strict=True):
        if i % 11 == 0:
            assert _all_nan(centroid)
        elif i % 7 == 0:
            assert centroid is None
        else:
            assert centroid == pytest.approx((i + 0.5, 0.5))


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"polygon": [1.0]}).select(geo.centroid("polygon"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


@pytest.mark.parametrize("anchor", ["length_weighted_mean", "vertex_mean"])
def test_the_namespace_matches_the_function(anchor: CentroidAnchor) -> None:
    df = _polygons([_L], [_SQUARE])

    assert df.select(pl.col("polygon").geo.centroid(anchor=anchor)).equals(
        df.select(geo.centroid("polygon", anchor=anchor))
    )


def test_rows_line_up_across_chunks_and_slices() -> None:
    square = _ring((10, 10), (12, 10), (12, 12), (10, 12))
    df = pl.concat(
        [_polygons([_SQUARE], [_L]), _polygons(None, [square], [_SQUARE])],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    got = _centroids(df.slice(1, 4))

    assert got == [
        pytest.approx((5 / 6, 5 / 6)),
        None,
        pytest.approx((11.0, 11.0)),
        pytest.approx((0.5, 0.5)),
    ]


# Linestrings

Line = list[tuple[float, ...]] | None


def _lines(
    *lines: Line, dtype: type[GeoArrowType] = LineStringXY, crs: str | None = None
) -> pl.DataFrame:
    """A `line` column of `dtype`, one row per list of coordinate tuples."""
    names = dtype._dimension
    vertex = pl.Struct(dict.fromkeys(names, pl.Float64))
    rows = [
        None if line is None else [dict(zip(names, c, strict=True)) for c in line]
        for line in lines
    ]
    storage = pl.Series("line", rows, dtype=pl.List(vertex))
    return pl.DataFrame(storage.ext.to(dtype(crs=crs)))


def _line_centroids(
    df: pl.DataFrame,
    method: CentroidMethod = "footprint",
    on_row_error: OnRowError = "raise",
) -> list[tuple[float, ...] | None]:
    out = (
        df.select(geo.centroid("line", method=method, on_row_error=on_row_error))
        .to_series()
        .ext.storage()
        .to_list()
    )
    return [None if c is None else tuple(c.values()) for c in out]


# Segments of length 2 and 1: its vertices average to (4/3, 1/3) instead.
_HOOK = [(0, 0), (2, 0), (2, 1)]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ([(0, 0), (2, 4)], (1.0, 2.0)),
        (_HOOK, (4 / 3, 1 / 6)),
        ([(0, 0), (1, 0), (1, 0), (2, 0)], (1.0, 0.0)),
    ],
    ids=["segment", "hook", "repeated vertex"],
)
def test_a_lines_centroid_is_its_midpoints_weighted_by_length(
    line: Line, expected: tuple[float, float]
) -> None:
    (got,) = _line_centroids(_lines(line))

    assert got == pytest.approx(expected)


# Segments of length 2 (straight up) and 1, of which only the second has length in x/y.
_RISING_HOOK = [(0, 0, 0), (0, 0, 2), (1, 0, 2)]


@pytest.mark.parametrize(
    ("method", "expected"),
    [("footprint", (0.5, 0.0, 2.0)), ("3d", (1 / 6, 0.0, 4 / 3))],
)
def test_the_method_says_whether_z_counts_towards_the_length(
    method: CentroidMethod, expected: tuple[float, float, float]
) -> None:
    (got,) = _line_centroids(_lines(_RISING_HOOK, dtype=LineStringXYZ), method)

    assert got == pytest.approx(expected)


def test_a_vertical_line_only_has_a_centroid_in_3d() -> None:
    df = _lines([(1, 1, 0), (1, 1, 5)], dtype=LineStringXYZ)

    assert _line_centroids(df, "3d") == [pytest.approx((1.0, 1.0, 2.5))]
    with pytest.raises(pl.exceptions.ComputeError, match='length with `method="3d"`'):
        _line_centroids(df)


@pytest.mark.parametrize("method", ["footprint", "3d"])
def test_without_z_both_methods_agree(method: CentroidMethod) -> None:
    lines = _lines(_HOOK, [(0, 0), (2, 4)])
    polygons = _polygons([_L], [_SQUARE])

    assert lines.select(geo.centroid("line", method=method)).equals(
        lines.select(geo.centroid("line"))
    )
    assert polygons.select(geo.centroid("polygon", method=method)).equals(
        polygons.select(geo.centroid("polygon"))
    )


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="`method` must be one of"):
        geo.centroid("line", method="3D")  # type: ignore[arg-type]


def test_an_unknown_anchor_is_refused() -> None:
    with pytest.raises(ValueError, match="`anchor` must be one of"):
        geo.centroid("line", anchor="first_vertex")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["footprint", "3d"])
def test_a_line_ignores_the_anchor(method: CentroidMethod) -> None:
    df = _lines(_RISING_HOOK, dtype=LineStringXYZ)

    assert df.select(geo.centroid("line", method=method, anchor="vertex_mean")).equals(
        df.select(geo.centroid("line", method=method))
    )


def test_m_is_averaged_along_the_length_without_adding_to_it() -> None:
    line = [(x, y, m) for (x, y), m in zip(_HOOK, [0, 10, 10], strict=True)]

    (got,) = _line_centroids(_lines(line, dtype=LineStringXYM))

    # m: (2 * 5 + 1 * 10) / 3
    assert got == pytest.approx((4 / 3, 1 / 6, 20 / 3))


def test_a_line_keeps_its_dimension(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.lines(line_coords).select(geo.centroid("line"))

    assert out.schema["line"] == dimension.point_dtype()


def test_the_crs_of_a_line_is_carried_over() -> None:
    df = _lines(_HOOK, crs="EPSG:4326")

    out = df.select(geo.centroid("line"))

    assert out.schema["line"] == PointXY._with_metadata_of(df.schema["line"])


def test_an_empty_line_has_the_empty_point_as_its_centroid() -> None:
    null, empty, hook = _line_centroids(_lines(None, [], _HOOK))

    assert null is None
    assert _all_nan(empty)
    assert hook == pytest.approx((4 / 3, 1 / 6))


@pytest.mark.parametrize(
    "line", [[(1, 1)], [(1, 1), (1, 1)]], ids=["one vertex", "one spot"]
)
def test_a_line_without_length_has_no_centroid(line: Line) -> None:
    df = _lines(line, _HOOK)

    with pytest.raises(pl.exceptions.ComputeError, match=_NO_LENGTH):
        _line_centroids(df)
    assert _line_centroids(df, on_row_error="null") == [
        None,
        pytest.approx((4 / 3, 1 / 6)),
    ]


def test_a_line_too_small_for_its_length_has_no_centroid() -> None:
    df = _lines([(0, 0), (1e-200, 1e-200)], _HOOK)

    with pytest.raises(pl.exceptions.ComputeError, match=_UNDERFLOW):
        _line_centroids(df)
    assert _line_centroids(df, on_row_error="null")[0] is None


@pytest.mark.parametrize("method", ["footprint", "3d"])
def test_the_namespace_matches_the_function_for_lines(method: CentroidMethod) -> None:
    df = _lines(_RISING_HOOK, dtype=LineStringXYZ)

    assert df.select(pl.col("line").geo.centroid(method=method)).equals(
        df.select(geo.centroid("line", method=method))
    )


def test_line_rows_line_up_across_chunks_and_slices() -> None:
    df = pl.concat(
        [_lines([(0, 0), (2, 4)], _HOOK), _lines(None, [(10, 10), (12, 10)], _HOOK)],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    got = _line_centroids(df.slice(1, 4))

    assert got == [
        pytest.approx((4 / 3, 1 / 6)),
        None,
        pytest.approx((11.0, 10.0)),
        pytest.approx((4 / 3, 1 / 6)),
    ]


# Multi-geometries: one centroid, of the parts weighted by their length or area


def _multilines(*multis: list[list[tuple[float, float]]] | None) -> pl.DataFrame:
    """A `multi` column of `MultiLineStringXY`, one row per list of parts."""
    rows = [
        None
        if multi is None
        else [[{"x": x, "y": y} for x, y in line] for line in multi]
        for multi in multis
    ]
    storage = pl.Series("multi", rows, dtype=pl.List(_XY_VERTICES))
    return pl.DataFrame(storage.ext.to(MultiLineStringXY()))


def _multipolygons(
    *multis: list[list[Ring]] | None, crs: str | None = None
) -> pl.DataFrame:
    """A `multi` column of `MultiPolygonXY`, one row per list of polygons."""
    storage = pl.Series("multi", list(multis), dtype=pl.List(_XY_RINGS))
    return pl.DataFrame(storage.ext.to(MultiPolygonXY(crs=crs)))


def _multi_centroids(
    df: pl.DataFrame, on_row_error: OnRowError = "raise"
) -> list[tuple[float, ...] | None]:
    out = (
        df.select(geo.centroid("multi", on_row_error=on_row_error))
        .to_series()
        .ext.storage()
        .to_list()
    )
    return [None if c is None else tuple(c.values()) for c in out]


def test_a_multilinestrings_parts_are_weighted_by_length() -> None:
    # 2 at (1, 0) and 1 at (10.5, 0)
    df = _multilines([[(0, 0), (2, 0)], [(10, 0), (11, 0)]])

    (got,) = _multi_centroids(df)

    assert got == pytest.approx((12.5 / 3, 0.0))


def test_a_multipolygons_parts_are_weighted_by_area() -> None:
    # 1 at (0.5, 0.5) and 4 at (11, 11)
    df = _multipolygons([[_SQUARE], [_BIG_SQUARE]])

    (got,) = _multi_centroids(df)

    assert got == pytest.approx((8.9, 8.9))


def test_a_part_without_length_adds_nothing() -> None:
    df = _multilines([[(0, 0), (2, 4)], [(1, 1)], [], [(5, 5), (5, 5)]])

    assert _multi_centroids(df) == [pytest.approx((1.0, 2.0))]


def test_a_part_without_area_adds_nothing() -> None:
    flat = [_ring((0, 0), (2, 0), (1, 0))]
    df = _multipolygons([[_BIG_SQUARE], [], flat])

    assert _multi_centroids(df) == [pytest.approx((11.0, 11.0))]


def test_an_empty_multilinestring_has_the_empty_point_as_its_centroid() -> None:
    null, empty, empty_parts = _multi_centroids(_multilines(None, [], [[], []]))

    assert null is None
    assert _all_nan(empty)
    assert _all_nan(empty_parts)


def test_a_multilinestring_without_length_has_no_centroid() -> None:
    df = _multilines([[(1, 1)], []])

    with pytest.raises(pl.exceptions.ComputeError, match=_NO_LENGTH):
        _multi_centroids(df)
    assert _multi_centroids(df, on_row_error="null") == [None]


def test_an_empty_multipolygon_has_the_empty_point_as_its_centroid() -> None:
    null, empty, empty_parts = _multi_centroids(_multipolygons(None, [], [[], [[]]]))

    assert null is None
    assert _all_nan(empty)
    assert _all_nan(empty_parts)


@pytest.mark.parametrize(
    ("polygons", "error"),
    [
        ([[_ring((0, 0), (2, 0), (1, 0))]], _NO_AREA),
        ([[_TINY_TRIANGLE]], _UNDERFLOW),
        ([[_SQUARE, _BIG_SQUARE]], _HOLES_EXCEED),
    ],
    ids=["flat", "tiny", "hole exceeds exterior"],
)
def test_a_multipolygon_without_area_has_no_centroid(
    polygons: list[list[Ring]], error: str
) -> None:
    df = _multipolygons(polygons)

    with pytest.raises(pl.exceptions.ComputeError, match=error):
        _multi_centroids(df)
    assert _multi_centroids(df, on_row_error="null") == [None]


# Non-finite coordinates: the centroid is not finite either, rather than null

_NAN = float("nan")
_INF = float("inf")


def _all_nan(centroid: tuple[float, ...] | None) -> bool:
    return centroid is not None and all(math.isnan(c) for c in centroid)


@pytest.mark.parametrize(
    "line",
    [[(0, 0), (_NAN, 1)], [(0, 1), (_INF, 1)], [(0, 0), (0, -_INF), (1, 0)]],
    ids=["nan", "inf", "-inf"],
)
def test_a_non_finite_line_has_a_nan_centroid(line: Line) -> None:
    (got,) = _line_centroids(_lines(line))

    assert _all_nan(got)


def test_a_nan_z_only_changes_the_z_of_a_footprint_centroid() -> None:
    (got,) = _line_centroids(_lines([(0, 0, 0), (2, 0, _NAN)], dtype=LineStringXYZ))

    assert got is not None
    assert got[:2] == pytest.approx((1.0, 0.0))
    assert math.isnan(got[2])


def test_a_nan_z_changes_the_3d_length_of_a_line() -> None:
    df = _lines([(0, 0, 0), (2, 0, _NAN)], dtype=LineStringXYZ)

    (got,) = _line_centroids(df, "3d")

    assert _all_nan(got)


def test_a_nan_m_only_changes_the_m_of_a_line() -> None:
    (got,) = _line_centroids(_lines([(0, 0, 0), (2, 0, _NAN)], dtype=LineStringXYM))

    assert got is not None
    assert got[:2] == pytest.approx((1.0, 0.0))
    assert math.isnan(got[2])


@pytest.mark.parametrize(
    "rings",
    [
        [_ring((0, 0), (1, 0), (_NAN, 1), (0, 1))],
        [_ring((0, 0), (1, 0), (_INF, 1), (0, 1))],
        # Its infinities add up, rather than cancel to NaN: the hole weighs -inf.
        [
            _ring((0, 0), (4, 0), (4, 4), (0, 4)),
            _ring((1, 1), (1, 2), (_INF, 6), (1, 0)),
        ],
    ],
    ids=["nan", "inf", "inf-in-hole"],
)
def test_a_non_finite_polygon_has_a_nan_centroid(rings: list[Ring]) -> None:
    (got,) = _centroids(_polygons(rings))

    assert _all_nan(got)


def test_a_nan_z_only_changes_the_z_of_a_polygon() -> None:
    out = _polygon_xyzm([_SQUARE], lambda x, y: _NAN if x == y == 1 else 0.0).select(
        geo.centroid("polygon")
    )

    (got,) = out.to_series().ext.storage().to_list()

    assert (got["x"], got["y"]) == pytest.approx((0.5, 0.5))
    assert math.isnan(got["z"])
    assert math.isnan(got["m"])


def test_a_nan_z_makes_the_3d_centroid_of_a_polygon_nan() -> None:
    df = _polygon_xyzm([_SQUARE], lambda x, y: _NAN if x == y == 1 else 0.0)

    (got,) = _centroids_of_only_column(df, "3d")

    assert _all_nan(got)


def test_a_nan_m_only_changes_the_m_of_a_3d_polygon_centroid() -> None:
    df = _polygon_xyzm(
        [_SQUARE], lambda _x, _y: 0.0, lambda x, y: _NAN if x == y == 1 else 0.0
    )

    (got,) = _centroids_of_only_column(df, "3d")

    assert got is not None
    assert got[:3] == pytest.approx((0.5, 0.5, 0.0))
    assert math.isnan(got[3])


def test_one_non_finite_part_makes_the_multi_centroid_nan() -> None:
    lines = _multilines([[(0, 0), (2, 0)], [(10, 0), (_NAN, 0)]])
    polygons = _multipolygons(
        [[_BIG_SQUARE], [_ring((0, 0), (1, 0), (_NAN, 1), (0, 1))]]
    )

    assert _all_nan(_multi_centroids(lines)[0])
    assert _all_nan(_multi_centroids(polygons)[0])


def _weighted_mean_of_parts(
    multis: pl.DataFrame,
    name: str,
    part: GeoArrowType,
    weight: Callable[[str], pl.Expr],
    method: CentroidMethod = "footprint",
) -> pl.DataFrame:
    """The centroid of each part on its own, weighted by `weight` and averaged
    per row, as plain float columns."""
    parts = (
        multis.with_row_index("row")
        .select("row", pl.col(name).ext.storage())
        .explode(name)
        .select("row", pl.col(name).ext.to(part))
    )
    centroids = parts.select(
        "row",
        weight(name).alias("weight"),
        geo.centroid(name, method=method).ext.storage().struct.unnest(),
    )
    fields = [c for c in centroids.columns if c not in ("row", "weight")]
    return (
        centroids.group_by("row", maintain_order=True)
        .agg(
            (pl.col(f) * pl.col("weight")).sum() / pl.col("weight").sum()
            for f in fields
        )
        .drop("row")
    )


_LENGTHS: dict[CentroidMethod, Callable[[str], pl.Expr]] = {
    "footprint": lambda name: geo.length(geo.force_2d(name), distance_method="planar"),
    "3d": lambda name: geo.length(name, distance_method="planar"),
}


@pytest.mark.parametrize("method", ["footprint", "3d"])
def test_a_multilinestring_weighs_each_part_like_a_lone_linestring(
    ring_coords: pl.DataFrame, dimension: Dimension, method: CentroidMethod
) -> None:
    multis = dimension.multilinestrings(ring_coords)
    out = multis.select(geo.centroid("multilinestring", method=method))
    expected = _weighted_mean_of_parts(
        multis,
        "multilinestring",
        dimension.linestring_dtype(),
        _LENGTHS[method],
        method,
    )

    assert out.schema["multilinestring"] == dimension.point_dtype()
    assert_frame_equal(coordinates(out, "multilinestring"), expected)


def test_a_multipolygon_weighs_each_part_like_a_lone_polygon(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    multis = dimension.multipolygons(multipolygon_coords)
    out = multis.select(geo.centroid("multipolygon"))
    expected = _weighted_mean_of_parts(
        multis,
        "multipolygon",
        dimension.polygon_dtype(),
        lambda g: geo.area(g, distance_method="planar"),
    )

    assert out.schema["multipolygon"] == dimension.point_dtype()
    assert_frame_equal(coordinates(out, "multipolygon"), expected)


def test_the_crs_of_a_multi_geometry_is_carried_over() -> None:
    df = _multipolygons([[_SQUARE]], crs="EPSG:4326")

    out = df.select(geo.centroid("multi"))

    assert out.schema["multi"] == PointXY._with_metadata_of(df.schema["multi"])


def test_multi_rows_line_up_across_chunks_and_slices() -> None:
    df = pl.concat(
        [
            _multipolygons([[_SQUARE]], [[_L], [_SQUARE]]),
            _multipolygons(None, [[_BIG_SQUARE]]),
        ],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    got = _multi_centroids(df.slice(1, 3))

    # 3 at (5/6, 5/6) and 1 at (0.5, 0.5)
    assert got == [pytest.approx((0.75, 0.75)), None, pytest.approx((11.0, 11.0))]


def test_the_namespace_matches_the_function_for_multi_geometries() -> None:
    df = _multilines([[(0, 0), (2, 4)], _HOOK])

    assert df.select(pl.col("multi").geo.centroid()).equals(
        df.select(geo.centroid("multi"))
    )


# Points: no length or area, so every point weighs the same


def test_a_point_is_its_own_centroid(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())
    out = df.select(geo.centroid("point"))

    assert out.schema["point"] == dimension.point_dtype()
    assert_frame_equal(coordinates(out), coordinates(df))


def test_a_multipoints_centroid_is_its_mean_coordinate(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)
    out = df.select(geo.centroid("multipoint"))

    assert out.schema["multipoint"] == dimension.point_dtype()
    assert out.equals(df.select(geo.mean_coordinate("multipoint")))


def test_an_empty_multipoint_has_the_empty_point_as_its_centroid() -> None:
    points = [[{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 4.0}], [], None]
    df = pl.DataFrame(
        pl.Series("multipoint", points, dtype=_XY_VERTICES).ext.to(MultiPointXY())
    )

    both, empty, null = _centroids_of_only_column(df, "footprint")

    assert both == (1.0, 2.0)
    assert _all_nan(empty)
    assert null is None


def test_the_crs_of_a_multipoint_is_carried_over() -> None:
    points = [{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 4.0}]
    df = pl.DataFrame({"points": [points]}, schema={"points": _XY_VERTICES}).select(
        geo.multi_point("points", crs="EPSG:4326").alias("multipoint")
    )

    out = df.select(geo.centroid("multipoint"))

    assert out.schema["multipoint"] == PointXY._with_metadata_of(
        df.schema["multipoint"]
    )
    assert coordinates(out, "multipoint").row(0) == (1.0, 2.0)
