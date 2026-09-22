"""Building `geoarrow.multilinestring` columns with `geometry.multilinestring`."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

from geopolars import geometry
from geopolars.datatypes import (
    GeoLineString,
    LineStringXY,
    MultiLineStringXY,
    MultiLineStringXYM,
)
from tests.unit.conftest import Dimension, multilinestring_coordinates

# A list of bare coordinate structs: what a linestring stores, before it is one.
_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

# A list of those: what a multilinestring stores, before it is one.
_XY_LINES = pl.List(_XY_VERTICES)

# One coordinate column per axis: one list per multilinestring, of one per part.
_XY_COORDS = {
    "lon": pl.List(pl.List(pl.Float64)),
    "lat": pl.List(pl.List(pl.Float64)),
}

# A single open two-vertex part, the smallest multilinestring that holds a line.
_SEGMENT = [[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]]


def test_the_linestrings_decide_the_dtype(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A multilinestring has the dimension of the linestrings it is built from."""
    df = dimension.multilinestrings(ring_coords)

    assert df.schema["multilinestring"] == dimension.multilinestring_dtype()


def test_accepts_bare_lists_of_coordinate_structs() -> None:
    """The parts do not have to be linestrings already:
    a list of the vertex lists a linestring wraps is the same storage."""
    df = pl.DataFrame({"lines": [_SEGMENT]}, schema={"lines": _XY_LINES}).select(
        geometry.multilinestring("lines").alias("multilinestring")
    )

    assert df.schema["multilinestring"] == MultiLineStringXY()
    assert_frame_equal(
        multilinestring_coordinates(df),
        pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}),
    )


def test_the_same_vertices_build_a_multilinestring_and_a_polygon(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    multilines = dimension.multilinestrings(ring_coords)
    polygons = dimension.polygons(ring_coords)

    assert_frame_equal(
        multilinestring_coordinates(multilines),
        multilinestring_coordinates(polygons, "polygon"),
    )
    assert multilines.schema["multilinestring"] != polygons.schema["polygon"]


def test_a_part_need_not_be_closed() -> None:
    """A polygon's rings have to close; a multilinestring's parts are lines,
    so an open one is left exactly as it came in."""
    df = pl.DataFrame({"lines": [_SEGMENT]}, schema={"lines": _XY_LINES}).select(
        geometry.multilinestring("lines").alias("multilinestring")
    )
    parts = pl.col("multilinestring").ext.storage()

    assert df.select(parts.explode(empty_as_null=False).list.len())[
        "multilinestring"
    ].to_list() == [2]


def test_an_empty_list_is_an_empty_multilinestring() -> None:
    """See https://geoarrow.org/format.html#empty-geometries"""
    df = pl.DataFrame({"lines": [[]]}, schema={"lines": _XY_LINES}).select(
        geometry.multilinestring("lines").alias("multilinestring")
    )

    assert df.schema["multilinestring"] == MultiLineStringXY()
    assert df["multilinestring"].is_null().to_list() == [False]
    assert df.select(pl.col("multilinestring").ext.storage().list.len())[
        "multilinestring"
    ].to_list() == [0]


def test_a_single_linestring_is_a_valid_multilinestring() -> None:
    df = pl.DataFrame({"lines": [_SEGMENT]}, schema={"lines": _XY_LINES}).select(
        geometry.multilinestring("lines").alias("multilinestring")
    )

    assert df.schema["multilinestring"] == MultiLineStringXY()
    assert df.schema["multilinestring"] != LineStringXY()


def test_a_missing_linestring_invalidates_the_whole_multilinestring() -> None:
    """GeoArrow allows nulls only at the outermost level.
    See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {"lines": [[*_SEGMENT, None], _SEGMENT, None]},
        schema={"lines": _XY_LINES},
    ).select(geometry.multilinestring("lines").alias("multilinestring"))

    assert df["multilinestring"].is_null().to_list() == [True, False, True]


def test_a_missing_vertex_invalidates_the_whole_multilinestring() -> None:
    df = pl.DataFrame(
        {"lines": [[[{"x": 1.0, "y": 2.0}, None]], _SEGMENT]},
        schema={"lines": _XY_LINES},
    ).select(geometry.multilinestring("lines").alias("multilinestring"))

    assert df["multilinestring"].is_null().to_list() == [True, False]


def test_a_missing_coordinate_invalidates_the_whole_multilinestring() -> None:
    df = pl.DataFrame(
        {"lines": [[[{"x": 1.0, "y": None}]], _SEGMENT]},
        schema={"lines": _XY_LINES},
    ).select(geometry.multilinestring("lines").alias("multilinestring"))

    assert df["multilinestring"].is_null().to_list() == [True, False]


def test_metadata_is_carried_over_from_the_linestrings() -> None:
    metadata = '{"edges":"spherical"}'
    spherical = GeoLineString.ext_from_params(
        "geoarrow.linestring", LineStringXY().ext_storage(), metadata
    )
    df = (
        pl.DataFrame({"vertices": _SEGMENT}, schema={"vertices": _XY_VERTICES})
        .select(line=pl.col("vertices").ext.to(spherical))
        .select(pl.col("line").implode())
        .select(geometry.multilinestring("line").alias("multilinestring"))
    )

    assert df.schema["multilinestring"].ext_metadata() == metadata
    assert df.schema["multilinestring"] == MultiLineStringXY.ext_from_params(
        "geoarrow.multilinestring", MultiLineStringXY().ext_storage(), metadata
    )


def test_rejects_linestrings_that_are_not_coordinates() -> None:
    df = pl.DataFrame({"lines": [[[1.0, 2.0]]]})

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.multilinestring("lines"))


def test_rejects_a_column_that_is_not_a_list() -> None:
    """A linestring column is a part per row, not a multilinestring per row."""
    df = pl.DataFrame({"vertices": _SEGMENT}, schema={"vertices": _XY_VERTICES}).select(
        geometry.linestring("vertices").alias("line")
    )

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.multilinestring("line"))


def test_rejects_a_list_of_points() -> None:
    """A multilinestring is built out of linestrings, and a point is not one.
    That gather is a multipoint."""
    df = (
        pl.DataFrame({"x": [1.0], "y": [2.0]})
        .select(geometry.point("x", "y").alias("point"))
        .select(pl.col("point").implode())
    )

    with pytest.raises(ComputeError, match="expected a list of `geoarrow.linestring`s"):
        df.select(geometry.multilinestring("point"))


def test_rejects_bad_linestrings_while_resolving_the_schema() -> None:
    """The dimension is read off the input's dtype,
    so parts that are not coordinates are a schema error."""
    lf = pl.LazyFrame({"river": ["rhine"]}).select(geometry.multilinestring("river"))

    with pytest.raises(ComputeError, match="expected a list of"):
        lf.collect_schema()


def test_empty_frame_keeps_its_dtype() -> None:
    """A zero-row build still produces a multilinestring column."""
    df = pl.DataFrame(schema={"lines": _XY_LINES}).select(
        geometry.multilinestring("lines").alias("multilinestring")
    )

    assert df.height == 0
    assert df.schema["multilinestring"] == MultiLineStringXY()


def test_measures_stay_per_vertex() -> None:
    """`m` belongs to a vertex, not to the part or the collection."""
    df = pl.DataFrame(
        {
            "river": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [3.0, 4.0, 5.0],
            "m": [7.0, 8.0, 9.0],
        }
    )
    multilines = (
        df.group_by("river", maintain_order=True)
        .agg(geometry.point("x", "y", m="m").alias("point"))
        .select(geometry.linestring("point").alias("line"))
        .select(pl.col("line").implode())
        .select(geometry.multilinestring("line").alias("multilinestring"))
    )

    assert multilines.schema["multilinestring"] == MultiLineStringXYM()
    assert_frame_equal(
        multilinestring_coordinates(multilines), df.select("x", "y", "m")
    )


def test_coordinate_columns_decide_the_dtype(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multilinestrings_from_coords(ring_coords)

    assert df.schema["multilinestring"] == dimension.multilinestring_dtype()


def test_coordinate_columns_build_the_same_multilinestrings(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    assert_frame_equal(
        dimension.multilinestrings_from_coords(ring_coords),
        dimension.multilinestrings(ring_coords),
    )


def test_coordinate_columns_keep_their_parts_in_order() -> None:
    """The nesting is the part structure, and it is read in the order given."""
    df = pl.DataFrame(
        {
            "lon": [[[0.0, 1.0], [2.0, 3.0, 4.0]]],
            "lat": [[[5.0, 6.0], [7.0, 8.0, 9.0]]],
        }
    ).select(geometry.multilinestring("lon", "lat").alias("multilinestring"))

    assert df.select(pl.col("multilinestring").ext.storage().list.len())[
        "multilinestring"
    ].to_list() == [2]
    assert_frame_equal(
        multilinestring_coordinates(df),
        pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0, 4.0],
                "y": [5.0, 6.0, 7.0, 8.0, 9.0],
            }
        ),
    )


def test_coordinate_columns_are_cast_to_f64() -> None:
    """Coordinates are doubles.
    integer columns are widened rather than refused."""
    df = pl.DataFrame({"lon": [[[1, 3]]], "lat": [[[2, 4]]]}).select(
        geometry.multilinestring("lon", "lat").alias("multilinestring")
    )

    assert df.schema["multilinestring"] == MultiLineStringXY()
    assert_frame_equal(
        multilinestring_coordinates(df),
        pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]}),
    )


def test_a_missing_coordinate_invalidates_the_multilinestring_of_coords() -> None:
    df = pl.DataFrame(
        {"lon": [[[1.0]], [[1.0]]], "lat": [[[None]], [[2.0]]]}, schema=_XY_COORDS
    ).select(geometry.multilinestring("lon", "lat").alias("multilinestring"))

    assert df["multilinestring"].is_null().to_list() == [True, False]


@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        # A part's vertex counts disagree.
        ([[[1.0, 3.0]]], [[[2.0]]]),
        # The multilinestrings disagree about how many parts they have.
        ([[[1.0], [3.0]]], [[[2.0]]]),
    ],
)
def test_rejects_coordinate_columns_that_nest_differently(
    lon: list[list[list[float]]], lat: list[list[list[float]]]
) -> None:
    df = pl.DataFrame({"lon": lon, "lat": lat}, schema=_XY_COORDS)

    with pytest.raises(ComputeError, match="do not nest the same way"):
        df.select(geometry.multilinestring("lon", "lat"))


def test_rejects_coordinate_columns_nested_only_once() -> None:
    """One list per geometry is a linestring's shape;
    a multilinestring needs the parts inside it too."""
    lf = pl.LazyFrame({"lon": [[1.0]], "lat": [[2.0]]}).select(
        geometry.multilinestring("lon", "lat")
    )

    with pytest.raises(
        ComputeError, match="lists of lists of f64, one per multilinestring"
    ):
        lf.collect_schema()


def test_rejects_a_measure_without_a_y_coordinate() -> None:
    with pytest.raises(TypeError, match="without a y coordinate"):
        geometry.multilinestring("lon", m="dist")


def test_empty_frame_of_coordinates_keeps_its_dtype() -> None:
    df = pl.DataFrame(schema=_XY_COORDS).select(
        geometry.multilinestring("lon", "lat").alias("multilinestring")
    )

    assert df.height == 0
    assert df.schema["multilinestring"] == MultiLineStringXY()


def test_one_argument_dispatches_to_the_linestring_form() -> None:
    df = pl.DataFrame({"lines": [_SEGMENT]}, schema={"lines": _XY_LINES})

    assert_frame_equal(
        df.select(geometry.multilinestring("lines")),
        df.select(geometry.multilinestring_from_linestrings("lines")),
    )


def test_coordinate_columns_dispatch_to_the_column_form() -> None:
    df = pl.DataFrame({"lon": [[[0.0, 1.0]]], "lat": [[[2.0, 3.0]]]}, schema=_XY_COORDS)

    assert_frame_equal(
        df.select(geometry.multilinestring("lon", "lat")),
        df.select(geometry.multilinestring_from_columns("lon", "lat")),
    )
