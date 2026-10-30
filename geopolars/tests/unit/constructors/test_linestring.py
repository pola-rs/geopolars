"""Building `geoarrow.linestring` columns with `geopolars.geo.line_string`."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import LineStringXY
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
    ).select(geo.line_string("vertices").alias("line"))

    assert df.schema["line"] == LineStringXY()
    assert_frame_equal(
        line_coordinates(df), pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


def test_vertices_keep_their_order() -> None:
    """A linestring is its vertices *in order*; reversing them is another line."""
    forwards = pl.DataFrame(
        {"vertices": [[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]]},
        schema={"vertices": _XY_VERTICES},
    ).select(geo.line_string("vertices").alias("line"))
    backwards = forwards.select(
        pl.col("line").ext.storage().list.reverse().alias("vertices")
    ).select(geo.line_string("vertices").alias("line"))

    assert_frame_equal(
        line_coordinates(backwards),
        line_coordinates(forwards).reverse(),
    )


def test_an_empty_list_is_an_empty_linestring() -> None:
    """Unlike a point, a linestring has a faithful empty representation.
    See https://geoarrow.org/format.html#empty-geometries"""
    df = pl.DataFrame({"vertices": [[]]}, schema={"vertices": _XY_VERTICES}).select(
        geo.line_string("vertices").alias("line")
    )

    assert df.schema["line"] == LineStringXY()
    assert df["line"].is_null().to_list() == [False]
    assert df.select(pl.col("line").ext.storage().list.len())["line"].to_list() == [0]


def test_a_null_vertex_nulls_the_whole_linestring() -> None:
    """GeoArrow allows nulls only at the outermost level, so a line with
    a null vertex is a null line.
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
    ).select(geo.line_string("vertices").alias("line"))

    assert df["line"].is_null().to_list() == [True, False, True]


def test_a_null_coordinate_nulls_the_whole_linestring() -> None:
    """The same rule one level down: a vertex that is present but has no `y` is
    not a vertex, so its line cannot stand either."""
    df = pl.DataFrame(
        {"vertices": [[{"x": 1.0, "y": None}], [{"x": 1.0, "y": 2.0}]]},
        schema={"vertices": _XY_VERTICES},
    ).select(geo.line_string("vertices").alias("line"))

    assert df["line"].is_null().to_list() == [True, False]


def test_rejects_vertices_that_are_not_coordinates() -> None:
    df = pl.DataFrame({"vertices": [[1.0, 2.0]]})

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geo.line_string("vertices"))


def test_rejects_a_column_that_is_not_a_list() -> None:
    """A point column is a coordinate per row, not a line per row."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(
        geo.point("x", "y").alias("point")
    )

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geo.line_string("point"))


def test_empty_frame_keeps_its_dtype() -> None:
    """A zero-row build still produces a linestring column, not a bare list."""
    df = pl.DataFrame(schema={"vertices": _XY_VERTICES}).select(
        geo.line_string("vertices").alias("line")
    )

    assert df.height == 0
    assert df.schema["line"] == LineStringXY()


def test_rejects_a_linestring_column() -> None:
    """A line of lines is not a linestring; that nesting is a polygon."""
    df = pl.DataFrame(
        {"vertices": [[{"x": 1.0, "y": 2.0}]]}, schema={"vertices": _XY_VERTICES}
    ).select(geo.line_string("vertices").alias("line"))

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geo.line_string("line"))


def test_a_multipoint_becomes_a_linestring_of_its_points() -> None:
    """A multipoint stores its points the way a linestring stores its vertices,
    so its points are used as they are, and its CRS is kept."""
    multipoint = (
        pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
        .select(point=geo.point("x", "y", crs="EPSG:4326").implode())
        .select(geo.multi_point_from_points("point").alias("multipoint"))
    )

    df = multipoint.select(geo.line_string("multipoint").alias("line"))

    assert df.schema["line"] == LineStringXY(crs="EPSG:4326")
    assert_frame_equal(
        line_coordinates(df), pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


def test_rejects_bad_vertices_while_resolving_the_schema() -> None:
    """The dimension is read off the input's dtype,
    so vertices that are not coordinates are a schema error."""
    lf = pl.LazyFrame({"route": ["north"]}).select(geo.line_string("route"))

    with pytest.raises(ComputeError, match="expected a list of"):
        lf.collect_schema()


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
        geo.line_string("lon", "lat").alias("line")
    )

    assert_frame_equal(
        line_coordinates(df), pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


def test_coordinate_columns_are_cast_to_f64() -> None:
    """Coordinates are doubles. integer columns are widened rather than refused."""
    df = pl.DataFrame({"lon": [[1, 3]], "lat": [[2, 4]]}).select(
        geo.line_string("lon", "lat").alias("line")
    )

    assert df.schema["line"] == LineStringXY()
    assert_frame_equal(
        line_coordinates(df), pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


def test_an_empty_coordinate_list_is_an_empty_linestring() -> None:
    df = pl.DataFrame({"lon": [[]], "lat": [[]]}, schema=_XY_COORDS).select(
        geo.line_string("lon", "lat").alias("line")
    )

    assert df["line"].is_null().to_list() == [False]
    assert df.select(pl.col("line").ext.storage().list.len())["line"].to_list() == [0]


def test_a_null_coordinate_list_is_a_null_linestring() -> None:
    df = pl.DataFrame(
        {"lon": [[1.0], None], "lat": [[2.0], None]}, schema=_XY_COORDS
    ).select(geo.line_string("lon", "lat").alias("line"))

    assert df["line"].is_null().to_list() == [False, True]


def test_a_null_coordinate_nulls_the_whole_linestring_of_coords() -> None:
    """The same rule as for vertices: a vertex with a null `y` is not a vertex,
    so its line cannot stand.
    See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {"lon": [[1.0, 3.0], [1.0]], "lat": [[2.0, None], [2.0]]}, schema=_XY_COORDS
    ).select(geo.line_string("lon", "lat").alias("line"))

    assert df["line"].is_null().to_list() == [True, False]


@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        # Different vertex counts: there is no line the coordinates agree on.
        ([[1.0, 3.0]], [[2.0]]),
        # One line null where the other is empty is the same disagreement.
        ([None], [[]]),
    ],
)
def test_rejects_coordinate_columns_that_nest_differently(
    lon: list[list[float] | None], lat: list[list[float] | None]
) -> None:
    df = pl.DataFrame({"lon": lon, "lat": lat}, schema=_XY_COORDS)

    with pytest.raises(ComputeError, match="do not nest the same way"):
        df.select(geo.line_string("lon", "lat"))


def test_rejects_flat_coordinate_columns() -> None:
    """One vertex per row is a point column; a linestring needs them grouped."""
    lf = pl.LazyFrame({"lon": [1.0], "lat": [2.0]}).select(
        geo.line_string("lon", "lat")
    )

    with pytest.raises(ComputeError, match="lists of f64, one per linestring"):
        lf.collect_schema()


def test_rejects_coordinate_columns_that_are_not_numbers() -> None:
    lf = pl.LazyFrame({"lon": [["1.0"]], "lat": [["2.0"]]}).select(
        geo.line_string("lon", "lat")
    )

    with pytest.raises(ComputeError, match="lists of f64"):
        lf.collect_schema()


def test_rejects_a_measure_without_a_y_coordinate() -> None:
    """One argument is a column of vertex lists, and that has no room for a
    `z` or an `m` alongside it."""
    with pytest.raises(TypeError, match="without a y coordinate"):
        geo.line_string("lon", m="dist")


def test_empty_frame_of_coordinates_keeps_its_dtype() -> None:
    df = pl.DataFrame(schema=_XY_COORDS).select(
        geo.line_string("lon", "lat").alias("line")
    )

    assert df.height == 0
    assert df.schema["line"] == LineStringXY()


def test_one_argument_dispatches_to_the_vertex_form() -> None:
    """`line_string` is a front door onto the two named constructors."""
    df = pl.DataFrame(
        {"vertices": [[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]]},
        schema={"vertices": _XY_VERTICES},
    )

    assert_frame_equal(
        df.select(geo.line_string("vertices")),
        df.select(geo.line_string_from_vertices("vertices")),
    )


def test_coordinate_columns_dispatch_to_the_column_form() -> None:
    df = pl.DataFrame({"lon": [[1.0, 3.0]], "lat": [[2.0, 4.0]], "ele": [[5.0, 6.0]]})

    assert_frame_equal(
        df.select(geo.line_string("lon", "lat", z="ele")),
        df.select(geo.line_string_from_columns("lon", "lat", z="ele")),
    )
