"""Building `geoarrow.linestring` columns with `geopolars.geometry.linestring`."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

from geopolars import geometry
from geopolars.datatypes import GeoPoint, LineStringXY, LineStringXYM, PointXY
from tests.unit.conftest import Dimension, line_coordinates

# A list of bare coordinate structs: what a linestring stores, before it is one.
_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

# One coordinate column per axis, grouped into one list per linestring.
_XY_COORDS = {"lon": pl.List(pl.Float64), "lat": pl.List(pl.Float64)}


def test_the_vertices_decide_the_dtype(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A linestring has the dimension of the points it is built from."""
    df = dimension.lines(line_coords)

    assert df.schema["line"] == dimension.linestring_dtype()


def test_accepts_bare_coordinate_structs() -> None:
    """The vertices do not have to be points already: a list of the coordinate
    structs a point wraps is the same storage, and is read the same way."""
    df = pl.DataFrame(
        {"vertices": [[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]]},
        schema={"vertices": _XY_VERTICES},
    ).select(geometry.linestring("vertices").alias("line"))

    assert df.schema["line"] == LineStringXY()
    assert_frame_equal(
        line_coordinates(df), pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


def test_vertices_keep_their_order() -> None:
    """A linestring is its vertices *in order*; reversing them is another line."""
    forwards = pl.DataFrame(
        {"vertices": [[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]]},
        schema={"vertices": _XY_VERTICES},
    ).select(geometry.linestring("vertices").alias("line"))
    backwards = forwards.select(
        pl.col("line").ext.storage().list.reverse().alias("vertices")
    ).select(geometry.linestring("vertices").alias("line"))

    assert_frame_equal(
        line_coordinates(backwards),
        line_coordinates(forwards).reverse(),
    )


def test_an_empty_list_is_an_empty_linestring() -> None:
    """Unlike a point, a linestring has a faithful empty representation.
    See https://geoarrow.org/format.html#empty-geometries"""
    df = pl.DataFrame({"vertices": [[]]}, schema={"vertices": _XY_VERTICES}).select(
        geometry.linestring("vertices").alias("line")
    )

    assert df.schema["line"] == LineStringXY()
    assert df["line"].is_null().to_list() == [False]
    assert df.select(pl.col("line").ext.storage().list.len())["line"].to_list() == [0]


def test_a_missing_vertex_invalidates_the_whole_linestring() -> None:
    """GeoArrow allows nulls only at the outermost level, so a line that is
    missing a vertex is a missing line.
    See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {
            "vertices": [
                [{"x": 1.0, "y": 2.0}, None],
                [{"x": 1.0, "y": 2.0}],
                None,
            ]
        },
        schema={"vertices": _XY_VERTICES},
    ).select(geometry.linestring("vertices").alias("line"))

    assert df["line"].is_null().to_list() == [True, False, True]


def test_a_missing_coordinate_invalidates_the_whole_linestring() -> None:
    """The same rule one level down: a vertex that is present but has no `y` is
    not a vertex, so its line cannot stand either."""
    df = pl.DataFrame(
        {"vertices": [[{"x": 1.0, "y": None}], [{"x": 1.0, "y": 2.0}]]},
        schema={"vertices": _XY_VERTICES},
    ).select(geometry.linestring("vertices").alias("line"))

    assert df["line"].is_null().to_list() == [True, False]


def test_metadata_is_carried_over_from_the_vertices() -> None:
    """The spec asks for `edges` to be propagated when an array is converted
    from one GeoArrow type to another, and points to linestrings is one.
    This version carries the whole metadata string, so `crs` comes along too."""
    metadata = '{"edges":"spherical"}'
    spherical = GeoPoint.ext_from_params(
        "geoarrow.point", PointXY().ext_storage(), metadata
    )
    df = (
        pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        .select(vertices=pl.struct("x", "y").ext.to(spherical))
        .select(pl.col("vertices").implode())
        .select(geometry.linestring("vertices").alias("line"))
    )

    assert df.schema["line"].ext_metadata() == metadata
    assert df.schema["line"] == LineStringXY.ext_from_params(
        "geoarrow.linestring", LineStringXY().ext_storage(), metadata
    )


def test_rejects_vertices_that_are_not_coordinates() -> None:
    df = pl.DataFrame({"vertices": [[1.0, 2.0]]})

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.linestring("vertices"))


def test_rejects_a_column_that_is_not_a_list() -> None:
    """A point column is a coordinate per row, not a line per row."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(
        geometry.point("x", "y").alias("point")
    )

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.linestring("point"))


def test_empty_frame_keeps_its_dtype() -> None:
    """A zero-row build still produces a linestring column, not a bare list."""
    df = pl.DataFrame(schema={"vertices": _XY_VERTICES}).select(
        geometry.linestring("vertices").alias("line")
    )

    assert df.height == 0
    assert df.schema["line"] == LineStringXY()


def test_measures_stay_per_vertex() -> None:
    """`m` belongs to a vertex, not to the line, so building a line out of
    points keeps one measure per vertex."""
    df = pl.DataFrame(
        {
            "id": [1, 1, 1],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "m": [7.0, 8.0, 9.0],
        }
    )
    lines = (
        df.group_by("id", maintain_order=True)
        .agg(geometry.point("x", "y", m="m").alias("point"))
        .select(geometry.linestring("point").alias("line"))
    )

    assert lines.schema["line"] == LineStringXYM()
    assert_frame_equal(line_coordinates(lines), df.select("x", "y", "m"))


def test_rejects_a_linestring_column() -> None:
    """A line of lines is not a linestring; that nesting is a polygon."""
    df = pl.DataFrame(
        {"vertices": [[{"x": 1.0, "y": 2.0}]]}, schema={"vertices": _XY_VERTICES}
    ).select(geometry.linestring("vertices").alias("line"))

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.linestring("line"))


def test_rejects_bad_vertices_while_resolving_the_schema() -> None:
    """The dimension is read off the input's dtype,
    so vertices that are not coordinates are a schema error."""
    lf = pl.LazyFrame({"route": ["north"]}).select(geometry.linestring("route"))

    with pytest.raises(ComputeError, match="expected a list of"):
        lf.collect_schema()


def test_coordinate_columns_decide_the_dtype(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.lines_from_coords(line_coords)

    assert df.schema["line"] == dimension.linestring_dtype()


def test_coordinate_columns_build_the_same_lines(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The two forms are two ways to spell one linestring, not two geometries."""
    assert_frame_equal(
        dimension.lines_from_coords(line_coords), dimension.lines(line_coords)
    )


def test_coordinate_columns_keep_their_vertices_in_order() -> None:
    """One vertex per position across the columns, in the order they are in."""
    df = pl.DataFrame({"lon": [[1.0, 3.0]], "lat": [[2.0, 4.0]]}).select(
        geometry.linestring("lon", "lat").alias("line")
    )

    assert_frame_equal(
        line_coordinates(df), pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


def test_coordinate_columns_are_cast_to_f64() -> None:
    """Coordinates are doubles. integer columns are widened rather than refused."""
    df = pl.DataFrame({"lon": [[1, 3]], "lat": [[2, 4]]}).select(
        geometry.linestring("lon", "lat").alias("line")
    )

    assert df.schema["line"] == LineStringXY()
    assert_frame_equal(
        line_coordinates(df), pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


def test_an_empty_coordinate_list_is_an_empty_linestring() -> None:
    df = pl.DataFrame({"lon": [[]], "lat": [[]]}, schema=_XY_COORDS).select(
        geometry.linestring("lon", "lat").alias("line")
    )

    assert df["line"].is_null().to_list() == [False]
    assert df.select(pl.col("line").ext.storage().list.len())["line"].to_list() == [0]


def test_a_missing_coordinate_list_is_a_missing_linestring() -> None:
    df = pl.DataFrame(
        {"lon": [[1.0], None], "lat": [[2.0], None]}, schema=_XY_COORDS
    ).select(geometry.linestring("lon", "lat").alias("line"))

    assert df["line"].is_null().to_list() == [False, True]


def test_a_missing_coordinate_invalidates_the_whole_linestring_of_coords() -> None:
    """The same rule as for vertices: a vertex missing its `y` is not a vertex,
    so its line cannot stand.
    See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {"lon": [[1.0, 3.0], [1.0]], "lat": [[2.0, None], [2.0]]}, schema=_XY_COORDS
    ).select(geometry.linestring("lon", "lat").alias("line"))

    assert df["line"].is_null().to_list() == [True, False]


@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        # Different vertex counts: there is no line the coordinates agree on.
        ([[1.0, 3.0]], [[2.0]]),
        # One line missing where the other is empty is the same disagreement.
        ([None], [[]]),
    ],
)
def test_rejects_coordinate_columns_that_nest_differently(
    lon: list[list[float] | None], lat: list[list[float] | None]
) -> None:
    df = pl.DataFrame({"lon": lon, "lat": lat}, schema=_XY_COORDS)

    with pytest.raises(ComputeError, match="do not nest the same way"):
        df.select(geometry.linestring("lon", "lat"))


def test_rejects_flat_coordinate_columns() -> None:
    """One vertex per row is a point column; a linestring needs them grouped."""
    lf = pl.LazyFrame({"lon": [1.0], "lat": [2.0]}).select(
        geometry.linestring("lon", "lat")
    )

    with pytest.raises(ComputeError, match="lists of f64, one per linestring"):
        lf.collect_schema()


def test_rejects_coordinate_columns_that_are_not_numbers() -> None:
    lf = pl.LazyFrame({"lon": [["1.0"]], "lat": [["2.0"]]}).select(
        geometry.linestring("lon", "lat")
    )

    with pytest.raises(ComputeError, match="lists of f64"):
        lf.collect_schema()


def test_rejects_a_measure_without_a_y_coordinate() -> None:
    """One argument is a column of vertex lists, and that has no room for a
    `z` or an `m` alongside it."""
    with pytest.raises(TypeError, match="without a y coordinate"):
        geometry.linestring("lon", m="dist")


def test_empty_frame_of_coordinates_keeps_its_dtype() -> None:
    df = pl.DataFrame(schema=_XY_COORDS).select(
        geometry.linestring("lon", "lat").alias("line")
    )

    assert df.height == 0
    assert df.schema["line"] == LineStringXY()


def test_one_argument_dispatches_to_the_vertex_form() -> None:
    """`linestring` is a front door onto the two named constructors."""
    df = pl.DataFrame(
        {"vertices": [[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]]},
        schema={"vertices": _XY_VERTICES},
    )

    assert_frame_equal(
        df.select(geometry.linestring("vertices")),
        df.select(geometry.linestring_from_vertices("vertices")),
    )


def test_coordinate_columns_dispatch_to_the_column_form() -> None:
    df = pl.DataFrame({"lon": [[1.0, 3.0]], "lat": [[2.0, 4.0]], "ele": [[5.0, 6.0]]})

    assert_frame_equal(
        df.select(geometry.linestring("lon", "lat", z="ele")),
        df.select(geometry.linestring_from_columns("lon", "lat", z="ele")),
    )
