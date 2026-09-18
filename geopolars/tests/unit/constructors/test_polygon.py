"""Building `geoarrow.polygon` columns with `geopolars.geometry.polygon`."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

from geopolars import geometry
from geopolars.datatypes import (
    GeoLineString,
    LineStringXY,
    PolygonXY,
    PolygonXYM,
)
from tests.unit.conftest import Dimension, ring_coordinates

# A list of bare coordinate structs: what a ring stores, before it is one.
_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

# A list of those: what a polygon stores, before it is one.
_XY_RINGS = pl.List(_XY_VERTICES)

# One coordinate column per axis: one list per polygon, of one list per ring.
_XY_COORDS = {
    "lon": pl.List(pl.List(pl.Float64)),
    "lat": pl.List(pl.List(pl.Float64)),
}

# One closed triangular ring, the smallest polygon that is a polygon.
_TRIANGLE = [
    [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.0},
        {"x": 0.0, "y": 1.0},
        {"x": 0.0, "y": 0.0},
    ]
]


def test_the_rings_decide_the_dtype(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon has the dimension of the rings it is built from."""
    df = dimension.polygons(ring_coords)

    assert df.schema["polygon"] == dimension.polygon_dtype()


def test_accepts_bare_lists_of_coordinate_structs() -> None:
    """The rings do not have to be linestrings already:
    a list of the vertex lists a linestring wraps is the same storage,
    and is read the same way."""
    df = pl.DataFrame({"rings": [_TRIANGLE]}, schema={"rings": _XY_RINGS}).select(
        geometry.polygon("rings").alias("polygon")
    )

    assert df.schema["polygon"] == PolygonXY()
    assert_frame_equal(
        ring_coordinates(df),
        pl.DataFrame({"x": [0.0, 1.0, 0.0, 0.0], "y": [0.0, 0.0, 1.0, 0.0]}),
    )


def test_rings_keep_their_order() -> None:
    """Which ring is which is positional:
    the first is the exterior ring and the rest are holes,
    so reordering them is another polygon."""
    hole = [{"x": 0.2, "y": 0.2}, {"x": 0.4, "y": 0.2}, {"x": 0.2, "y": 0.2}]
    df = pl.DataFrame(
        {"rings": [[*_TRIANGLE, hole]]}, schema={"rings": _XY_RINGS}
    ).select(geometry.polygon("rings").alias("polygon"))
    swapped = df.select(
        pl.col("polygon").ext.storage().list.reverse().alias("rings")
    ).select(geometry.polygon("rings").alias("polygon"))

    assert_frame_equal(
        ring_coordinates(swapped),
        pl.DataFrame(
            {
                "x": [0.2, 0.4, 0.2, 0.0, 1.0, 0.0, 0.0],
                "y": [0.2, 0.2, 0.2, 0.0, 0.0, 1.0, 0.0],
            }
        ),
    )


def test_an_empty_list_is_an_empty_polygon() -> None:
    """A polygon with no rings is a geometry this layout can hold,
    and is not a missing polygon.
    See https://geoarrow.org/format.html#empty-geometries"""
    df = pl.DataFrame({"rings": [[]]}, schema={"rings": _XY_RINGS}).select(
        geometry.polygon("rings").alias("polygon")
    )

    assert df.schema["polygon"] == PolygonXY()
    assert df["polygon"].is_null().to_list() == [False]
    assert df.select(pl.col("polygon").ext.storage().list.len())[
        "polygon"
    ].to_list() == [0]


def test_a_missing_ring_invalidates_the_whole_polygon() -> None:
    """GeoArrow allows nulls only at the outermost level,
    so a polygon that is missing a ring is a missing polygon.
    See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {"rings": [[*_TRIANGLE, None], _TRIANGLE, None]},
        schema={"rings": _XY_RINGS},
    ).select(geometry.polygon("rings").alias("polygon"))

    assert df["polygon"].is_null().to_list() == [True, False, True]


def test_a_missing_vertex_invalidates_the_whole_polygon() -> None:
    """The same rule one level further down than for a linestring:
    a ring missing a vertex cannot stand,
    and neither can the polygon around it."""
    df = pl.DataFrame(
        {"rings": [[[{"x": 1.0, "y": 2.0}, None]], _TRIANGLE]},
        schema={"rings": _XY_RINGS},
    ).select(geometry.polygon("rings").alias("polygon"))

    assert df["polygon"].is_null().to_list() == [True, False]


def test_a_missing_coordinate_invalidates_the_whole_polygon() -> None:
    """And one level down again:
    a vertex that is present but has no `y` is not a vertex,
    so its ring cannot stand, so its polygon cannot either."""
    df = pl.DataFrame(
        {"rings": [[[{"x": 1.0, "y": None}]], _TRIANGLE]},
        schema={"rings": _XY_RINGS},
    ).select(geometry.polygon("rings").alias("polygon"))

    assert df["polygon"].is_null().to_list() == [True, False]


def test_metadata_is_carried_over_from_the_rings() -> None:
    metadata = '{"edges":"spherical"}'
    spherical = GeoLineString.ext_from_params(
        "geoarrow.linestring", LineStringXY().ext_storage(), metadata
    )
    df = (
        pl.DataFrame({"vertices": _TRIANGLE}, schema={"vertices": _XY_VERTICES})
        .select(ring=pl.col("vertices").ext.to(spherical))
        .select(pl.col("ring").implode())
        .select(geometry.polygon("ring").alias("polygon"))
    )

    assert df.schema["polygon"].ext_metadata() == metadata
    assert df.schema["polygon"] == PolygonXY.ext_from_params(
        "geoarrow.polygon", PolygonXY().ext_storage(), metadata
    )


def test_rejects_rings_that_are_not_coordinates() -> None:
    df = pl.DataFrame({"rings": [[[1.0, 2.0]]]})

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.polygon("rings"))


def test_rejects_a_column_that_is_not_a_list() -> None:
    """A linestring column is a ring per row, not a polygon per row."""
    df = pl.DataFrame(
        {"vertices": _TRIANGLE}, schema={"vertices": _XY_VERTICES}
    ).select(geometry.linestring("vertices").alias("line"))

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.polygon("line"))


def test_rejects_a_list_of_vertices() -> None:
    """a list of coordinates is a ring, not a list of rings."""
    df = pl.DataFrame({"vertices": _TRIANGLE}, schema={"vertices": _XY_VERTICES})

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.polygon("vertices"))


def test_rejects_a_list_of_points() -> None:
    """A polygon is built out of rings, and a point is not a ring."""
    df = (
        pl.DataFrame({"x": [1.0], "y": [2.0]})
        .select(geometry.point("x", "y").alias("point"))
        .select(pl.col("point").implode())
    )

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.polygon("point"))


def test_empty_frame_keeps_its_dtype() -> None:
    """A zero-row build still produces a polygon column, not a bare list."""
    df = pl.DataFrame(schema={"rings": _XY_RINGS}).select(
        geometry.polygon("rings").alias("polygon")
    )

    assert df.height == 0
    assert df.schema["polygon"] == PolygonXY()


def test_measures_stay_per_vertex() -> None:
    """`m` belongs to a vertex, not to the ring or the polygon,
    so building a polygon out of linestrings keeps one measure per vertex."""
    df = pl.DataFrame(
        {
            "plot": [1, 1, 1, 1],
            "x": [0.0, 1.0, 0.0, 0.0],
            "y": [0.0, 0.0, 1.0, 0.0],
            "m": [7.0, 8.0, 9.0, 7.0],
        }
    )
    polygons = (
        df.group_by("plot", maintain_order=True)
        .agg(geometry.point("x", "y", m="m").alias("point"))
        .select(geometry.linestring("point").alias("line"))
        .select(pl.col("line").implode())
        .select(geometry.polygon("line").alias("polygon"))
    )

    assert polygons.schema["polygon"] == PolygonXYM()
    assert_frame_equal(ring_coordinates(polygons), df.select("x", "y", "m"))


def test_rejects_bad_rings_while_resolving_the_schema() -> None:
    """The dimension is read off the input's dtype,
    so rings that are not coordinates are a schema error."""
    lf = pl.LazyFrame({"plot": ["north"]}).select(geometry.polygon("plot"))

    with pytest.raises(ComputeError, match="expected a list of"):
        lf.collect_schema()


def test_coordinate_columns_decide_the_dtype(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons_from_coords(ring_coords)

    assert df.schema["polygon"] == dimension.polygon_dtype()


def test_coordinate_columns_build_the_same_polygons(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The two forms are two ways to spell one polygon, not two geometries."""
    assert_frame_equal(
        dimension.polygons_from_coords(ring_coords), dimension.polygons(ring_coords)
    )


def test_coordinate_columns_keep_their_rings_in_order() -> None:
    """The nesting is the ring structure: the first list is the exterior ring."""
    df = pl.DataFrame(
        {
            "lon": [[[0.0, 1.0, 0.0, 0.0], [0.2, 0.4, 0.2]]],
            "lat": [[[0.0, 0.0, 1.0, 0.0], [0.2, 0.2, 0.2]]],
        }
    ).select(geometry.polygon("lon", "lat").alias("polygon"))

    assert df.select(pl.col("polygon").ext.storage().list.len())[
        "polygon"
    ].to_list() == [2]
    assert_frame_equal(
        ring_coordinates(df),
        pl.DataFrame(
            {
                "x": [0.0, 1.0, 0.0, 0.0, 0.2, 0.4, 0.2],
                "y": [0.0, 0.0, 1.0, 0.0, 0.2, 0.2, 0.2],
            }
        ),
    )


def test_a_missing_coordinate_invalidates_the_whole_polygon_of_coords() -> None:
    """A vertex missing its `y` cannot stand, so neither can the ring around
    it, nor the polygon around that."""
    df = pl.DataFrame(
        {"lon": [[[1.0]], [[1.0]]], "lat": [[[None]], [[2.0]]]}, schema=_XY_COORDS
    ).select(geometry.polygon("lon", "lat").alias("polygon"))

    assert df["polygon"].is_null().to_list() == [True, False]


@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        # A ring's vertex counts disagree.
        ([[[1.0, 3.0]]], [[[2.0]]]),
        # The polygons disagree about how many rings they have.
        ([[[1.0], [3.0]]], [[[2.0]]]),
    ],
)
def test_rejects_coordinate_columns_that_nest_differently(
    lon: list[list[list[float]]], lat: list[list[list[float]]]
) -> None:
    df = pl.DataFrame({"lon": lon, "lat": lat}, schema=_XY_COORDS)

    with pytest.raises(ComputeError, match="do not nest the same way"):
        df.select(geometry.polygon("lon", "lat"))


def test_rejects_coordinate_columns_nested_only_once() -> None:
    """One list per geometry is a linestring's shape; a polygon needs the
    rings inside it too."""
    lf = pl.LazyFrame({"lon": [[1.0]], "lat": [[2.0]]}).select(
        geometry.polygon("lon", "lat")
    )

    with pytest.raises(ComputeError, match="lists of lists of f64, one per polygon"):
        lf.collect_schema()


def test_rejects_a_measure_without_a_y_coordinate() -> None:
    with pytest.raises(TypeError, match="without a y coordinate"):
        geometry.polygon("lon", m="dist")


def test_empty_frame_of_coordinates_keeps_its_dtype() -> None:
    df = pl.DataFrame(schema=_XY_COORDS).select(
        geometry.polygon("lon", "lat").alias("polygon")
    )

    assert df.height == 0
    assert df.schema["polygon"] == PolygonXY()


def test_one_argument_dispatches_to_the_ring_form() -> None:
    """`polygon` is a front door onto the two named constructors."""
    df = pl.DataFrame({"rings": [_TRIANGLE]}, schema={"rings": _XY_RINGS})

    assert_frame_equal(
        df.select(geometry.polygon("rings")),
        df.select(geometry.polygon_from_rings("rings")),
    )


def test_coordinate_columns_dispatch_to_the_column_form() -> None:
    df = pl.DataFrame(
        {"lon": [[[0.0, 1.0, 0.0]]], "lat": [[[0.0, 0.0, 1.0]]]}, schema=_XY_COORDS
    )

    assert_frame_equal(
        df.select(geometry.polygon("lon", "lat")),
        df.select(geometry.polygon_from_columns("lon", "lat")),
    )
