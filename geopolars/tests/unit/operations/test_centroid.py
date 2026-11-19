"""The centroid of a geometry: the centre of its length or area."""

from __future__ import annotations

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
    MultiPolygonXY,
    PointXY,
    PolygonXYZM,
)
from tests.unit.conftest import XY, Dimension, coordinates

if TYPE_CHECKING:
    from collections.abc import Callable

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)

Ring = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Ring:
    """A closed ring through `xy`."""
    return [{"x": x, "y": y} for x, y in (*xy, xy[0])]


_SQUARE = _ring((0, 0), (1, 0), (1, 1), (0, 1))
# An L of three unit squares: its vertices average elsewhere than its area.
_L = _ring((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))


def _polygons(*polygons: list[Ring] | None, crs: str | None = None) -> pl.DataFrame:
    """A polygon column, one row per list of XY rings."""
    return pl.DataFrame({"rings": list(polygons)}, schema={"rings": _XY_RINGS}).select(
        geo.polygons_from_rings_in_single_column("rings", crs=crs).alias("polygon")
    )


def _centroids(df: pl.DataFrame) -> list[tuple[float, float] | None]:
    out = df.select(geo.centroid("polygon")).to_series().ext.storage().to_list()
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
    rings: list[Ring], z: Callable[[float, float], float]
) -> pl.DataFrame:
    """A `PolygonXYZM` column of one polygon: `rings` with both z and m at `z(x, y)`."""
    rings_zm = [
        [{**c, "z": z(c["x"], c["y"]), "m": z(c["x"], c["y"])} for c in ring]
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
def test_z_and_m_are_averaged_over_the_area(
    rings: list[Ring], z: Callable[[float, float], float], expected: float
) -> None:
    # On a tilted plane, the centroid lies on it: above the centroid of the footprint.
    out = _polygon_xyzm(rings, z).select(geo.centroid("polygon"))

    (got,) = out.to_series().ext.storage().to_list()

    assert (got["z"], got["m"]) == pytest.approx((expected, expected))


def test_the_crs_is_carried_over() -> None:
    df = _polygons([_SQUARE], crs="EPSG:4326")

    out = df.select(geo.centroid("polygon"))

    assert out.schema["polygon"] == PointXY._with_metadata_of(df.schema["polygon"])


def test_a_null_or_empty_polygon_has_no_centroid() -> None:
    assert _centroids(_polygons(None, [], [_SQUARE])) == [None, None, (0.5, 0.5)]


def test_a_polygon_that_encloses_nothing_has_no_centroid() -> None:
    assert _centroids(_polygons([_ring((0, 0), (2, 0), (1, 0))])) == [None]


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"polygon": [1.0]}).select(geo.centroid("polygon"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


def test_the_namespace_matches_the_function() -> None:
    df = _polygons([_L], [_SQUARE])

    assert df.select(pl.col("polygon").geo.centroid()).equals(
        df.select(geo.centroid("polygon"))
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


def _line_centroids(df: pl.DataFrame) -> list[tuple[float, ...] | None]:
    out = df.select(geo.centroid("line")).to_series().ext.storage().to_list()
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


def test_z_counts_towards_the_length() -> None:
    # Segments of length 2 (straight up) and 1: in 2D the first would weigh nothing.
    line = [(0, 0, 0), (0, 0, 2), (1, 0, 2)]

    (got,) = _line_centroids(_lines(line, dtype=LineStringXYZ))

    assert got == pytest.approx((1 / 6, 0.0, 4 / 3))


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


def test_a_line_without_length_has_no_centroid() -> None:
    df = _lines(None, [], [(1, 1)], [(1, 1), (1, 1)], _HOOK)

    assert _line_centroids(df) == [
        None,
        None,
        None,
        None,
        pytest.approx((4 / 3, 1 / 6)),
    ]


def test_the_namespace_matches_the_function_for_lines() -> None:
    df = _lines(_HOOK, [(0, 0), (2, 4)])

    assert df.select(pl.col("line").geo.centroid()).equals(
        df.select(geo.centroid("line"))
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


def _multi_centroids(df: pl.DataFrame) -> list[tuple[float, ...] | None]:
    out = df.select(geo.centroid("multi")).to_series().ext.storage().to_list()
    return [None if c is None else tuple(c.values()) for c in out]


_BIG_SQUARE = _ring((10, 10), (12, 10), (12, 12), (10, 12))


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


def test_a_multilinestring_without_length_has_no_centroid() -> None:
    df = _multilines(None, [], [[(1, 1)], []])

    assert _multi_centroids(df) == [None, None, None]


def test_a_multipolygon_without_area_has_no_centroid() -> None:
    df = _multipolygons(None, [], [[]])

    assert _multi_centroids(df) == [None, None, None]


def _weighted_mean_of_parts(
    multis: pl.DataFrame,
    name: str,
    part: GeoArrowType,
    weight: Callable[[str], pl.Expr],
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
        geo.centroid(name).ext.storage().struct.unnest(),
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


def test_a_multilinestring_weighs_each_part_like_a_lone_linestring(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    multis = dimension.multilinestrings(ring_coords)
    out = multis.select(geo.centroid("multilinestring"))
    expected = _weighted_mean_of_parts(
        multis,
        "multilinestring",
        dimension.linestring_dtype(),
        lambda name: geo.length(name, method="planar"),
    )

    assert out.schema["multilinestring"] == dimension.point_dtype()
    assert_frame_equal(coordinates(out, "multilinestring"), expected)


def test_a_multipolygon_weighs_each_part_like_a_lone_polygon(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    multis = dimension.multipolygons(multipolygon_coords)
    out = multis.select(geo.centroid("multipolygon"))
    expected = _weighted_mean_of_parts(
        multis, "multipolygon", dimension.polygon_dtype(), geo.area_planar
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
