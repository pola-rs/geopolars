"""Building `geoarrow.multipoint` columns with `geopolars.geometry.multipoint`."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

from geopolars import geometry
from geopolars.datatypes import GeoPoint, MultiPointXY, MultiPointXYM, PointXY
from tests.unit.conftest import Dimension, multipoint_coordinates

# A list of bare coordinate structs: what a multipoint stores, before it is one.
_XY_POINTS = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

# One coordinate column per axis, grouped into one list per multipoint.
_XY_COORDS = {"lon": pl.List(pl.Float64), "lat": pl.List(pl.Float64)}


def test_the_points_decide_the_dtype(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A multipoint has the dimension of the points it is built from."""
    df = dimension.multipoints(line_coords)

    assert df.schema["multipoint"] == dimension.multipoint_dtype()


def test_accepts_bare_coordinate_structs() -> None:
    df = pl.DataFrame(
        {"points": [[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]]},
        schema={"points": _XY_POINTS},
    ).select(geometry.multipoint("points").alias("multipoint"))

    assert df.schema["multipoint"] == MultiPointXY()
    assert_frame_equal(
        multipoint_coordinates(df), pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


def test_the_same_points_build_a_multipoint_and_a_linestring(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    multipoints = dimension.multipoints(line_coords)
    lines = dimension.lines(line_coords)

    assert_frame_equal(
        multipoint_coordinates(multipoints),
        multipoint_coordinates(lines, "line"),
    )
    assert multipoints.schema["multipoint"] != lines.schema["line"]


def test_an_empty_list_is_an_empty_multipoint() -> None:
    """See https://geoarrow.org/format.html#empty-geometries"""
    df = pl.DataFrame({"points": [[]]}, schema={"points": _XY_POINTS}).select(
        geometry.multipoint("points").alias("multipoint")
    )

    assert df.schema["multipoint"] == MultiPointXY()
    assert df["multipoint"].is_null().to_list() == [False]
    assert df.select(pl.col("multipoint").ext.storage().list.len())[
        "multipoint"
    ].to_list() == [0]


def test_a_single_point_is_a_valid_multipoint() -> None:
    df = pl.DataFrame(
        {"points": [[{"x": 1.0, "y": 2.0}]]}, schema={"points": _XY_POINTS}
    ).select(geometry.multipoint("points").alias("multipoint"))

    assert df.schema["multipoint"] == MultiPointXY()
    assert df.schema["multipoint"] != PointXY()


def test_a_missing_point_invalidates_the_whole_multipoint() -> None:
    """GeoArrow allows nulls only at the outermost level,
    so a multipoint that is missing one of its points is a missing multipoint.
    See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {
            "points": [
                [{"x": 1.0, "y": 2.0}, None],
                [{"x": 1.0, "y": 2.0}],
                None,
            ]
        },
        schema={"points": _XY_POINTS},
    ).select(geometry.multipoint("points").alias("multipoint"))

    assert df["multipoint"].is_null().to_list() == [True, False, True]


def test_a_missing_coordinate_invalidates_the_whole_multipoint() -> None:
    """A missing 'y' in nesting should invalidate the multipoint."""
    df = pl.DataFrame(
        {"points": [[{"x": 1.0, "y": None}], [{"x": 1.0, "y": 2.0}]]},
        schema={"points": _XY_POINTS},
    ).select(geometry.multipoint("points").alias("multipoint"))

    assert df["multipoint"].is_null().to_list() == [True, False]


def test_metadata_is_carried_over_from_the_points() -> None:
    metadata = '{"edges":"spherical"}'
    spherical = GeoPoint.ext_from_params(
        "geoarrow.point", PointXY().ext_storage(), metadata
    )
    df = (
        pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        .select(points=pl.struct("x", "y").ext.to(spherical))
        .select(pl.col("points").implode())
        .select(geometry.multipoint("points").alias("multipoint"))
    )

    assert df.schema["multipoint"].ext_metadata() == metadata
    assert df.schema["multipoint"] == MultiPointXY.ext_from_params(
        "geoarrow.multipoint", MultiPointXY().ext_storage(), metadata
    )


def test_rejects_points_that_are_not_coordinates() -> None:
    df = pl.DataFrame({"points": [[1.0, 2.0]]})

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.multipoint("points"))


def test_rejects_a_column_that_is_not_a_list() -> None:
    """A point column is a coordinate per row, not a multipoint per row."""
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(
        geometry.point("x", "y").alias("point")
    )

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.multipoint("point"))


def test_rejects_a_multipoint_column() -> None:
    """A multipoint of multipoints is not a multipoint;
    that nesting is a multilinestring."""
    df = pl.DataFrame(
        {"points": [[{"x": 1.0, "y": 2.0}]]}, schema={"points": _XY_POINTS}
    ).select(geometry.multipoint("points").alias("multipoint"))

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geometry.multipoint("multipoint"))


def test_rejects_a_list_of_linestrings() -> None:
    lines = pl.DataFrame(
        {"vertices": [[{"x": 1.0, "y": 2.0}]]},
        schema={"vertices": pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))},
    ).select(geometry.linestring("vertices").alias("line"))

    with pytest.raises(ComputeError, match="expected a list of `geoarrow.point`s"):
        lines.select(pl.col("line").implode().alias("parts")).select(
            geometry.multipoint("parts")
        )


def test_rejects_bad_points_while_resolving_the_schema() -> None:
    """The dimension is read off the input's dtype,
    so points that are not coordinates are a schema error."""
    lf = pl.LazyFrame({"survey": ["north"]}).select(geometry.multipoint("survey"))

    with pytest.raises(ComputeError, match="expected a list of"):
        lf.collect_schema()


def test_empty_frame_keeps_its_dtype() -> None:
    """A zero-row build still produces a multipoint column, not a bare list."""
    df = pl.DataFrame(schema={"points": _XY_POINTS}).select(
        geometry.multipoint("points").alias("multipoint")
    )

    assert df.height == 0
    assert df.schema["multipoint"] == MultiPointXY()


def test_measures_stay_per_point() -> None:
    """`m` belongs to a point, not to the collection"""
    df = pl.DataFrame(
        {
            "id": [1, 1, 1],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "m": [7.0, 8.0, 9.0],
        }
    )
    multipoints = (
        df.group_by("id", maintain_order=True)
        .agg(geometry.point("x", "y", m="m").alias("point"))
        .select(geometry.multipoint("point").alias("multipoint"))
    )

    assert multipoints.schema["multipoint"] == MultiPointXYM()
    assert_frame_equal(multipoint_coordinates(multipoints), df.select("x", "y", "m"))


def test_coordinate_columns_decide_the_dtype(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints_from_coords(line_coords)

    assert df.schema["multipoint"] == dimension.multipoint_dtype()


def test_coordinate_columns_build_the_same_multipoints(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    assert_frame_equal(
        dimension.multipoints_from_coords(line_coords),
        dimension.multipoints(line_coords),
    )


def test_coordinate_columns_pair_up_by_position() -> None:
    """One point per position across the columns. A multipoint is unordered as
    a geometry, but the columns still have to be read in lockstep, or the x of
    one point lands on the y of another."""
    df = pl.DataFrame({"lon": [[1.0, 3.0]], "lat": [[2.0, 4.0]]}).select(
        geometry.multipoint("lon", "lat").alias("multipoint")
    )

    assert_frame_equal(
        multipoint_coordinates(df), pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


def test_coordinate_columns_are_cast_to_f64() -> None:
    """Coordinates are doubles. integer columns are widened rather than refused."""
    df = pl.DataFrame({"lon": [[1, 3]], "lat": [[2, 4]]}).select(
        geometry.multipoint("lon", "lat").alias("multipoint")
    )

    assert df.schema["multipoint"] == MultiPointXY()
    assert_frame_equal(
        multipoint_coordinates(df), pl.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]})
    )


def test_an_empty_coordinate_list_is_an_empty_multipoint() -> None:
    df = pl.DataFrame({"lon": [[]], "lat": [[]]}, schema=_XY_COORDS).select(
        geometry.multipoint("lon", "lat").alias("multipoint")
    )

    assert df["multipoint"].is_null().to_list() == [False]
    assert df.select(pl.col("multipoint").ext.storage().list.len())[
        "multipoint"
    ].to_list() == [0]


def test_a_missing_coordinate_list_is_a_missing_multipoint() -> None:
    """See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {"lon": [[1.0], None], "lat": [[2.0], None]}, schema=_XY_COORDS
    ).select(geometry.multipoint("lon", "lat").alias("multipoint"))

    assert df["multipoint"].is_null().to_list() == [False, True]


def test_a_missing_coordinate_invalidates_the_whole_multipoint_of_coords() -> None:
    """See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {"lon": [[1.0, 3.0], [1.0]], "lat": [[2.0, None], [2.0]]}, schema=_XY_COORDS
    ).select(geometry.multipoint("lon", "lat").alias("multipoint"))

    assert df["multipoint"].is_null().to_list() == [True, False]


@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        # Different point counts: there is no multipoint the coordinates agree on.
        ([[1.0, 3.0]], [[2.0]]),
        # One missing where the other is empty is the same disagreement.
        ([None], [[]]),
    ],
)
def test_rejects_coordinate_columns_that_nest_differently(
    lon: list[list[float] | None], lat: list[list[float] | None]
) -> None:
    df = pl.DataFrame({"lon": lon, "lat": lat}, schema=_XY_COORDS)

    with pytest.raises(ComputeError, match="do not nest the same way"):
        df.select(geometry.multipoint("lon", "lat"))


def test_rejects_flat_coordinate_columns() -> None:
    """One coordinate per row is a point column.
    a multipoint needs them grouped."""
    lf = pl.LazyFrame({"lon": [1.0], "lat": [2.0]}).select(
        geometry.multipoint("lon", "lat")
    )

    with pytest.raises(ComputeError, match="lists of f64, one per multipoint"):
        lf.collect_schema()


def test_rejects_coordinate_columns_that_are_not_numbers() -> None:
    lf = pl.LazyFrame({"lon": [["1.0"]], "lat": [["2.0"]]}).select(
        geometry.multipoint("lon", "lat")
    )

    with pytest.raises(ComputeError, match="lists of f64"):
        lf.collect_schema()


def test_empty_frame_of_coordinates_keeps_its_dtype() -> None:
    df = pl.DataFrame(schema=_XY_COORDS).select(
        geometry.multipoint("lon", "lat").alias("multipoint")
    )

    assert df.height == 0
    assert df.schema["multipoint"] == MultiPointXY()


def test_one_argument_dispatches_to_the_point_form() -> None:
    df = pl.DataFrame(
        {"points": [[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]]},
        schema={"points": _XY_POINTS},
    )

    assert_frame_equal(
        df.select(geometry.multipoint("points")),
        df.select(geometry.multipoint_from_points("points")),
    )


def test_coordinate_columns_dispatch_to_the_column_form() -> None:
    df = pl.DataFrame({"lon": [[1.0, 3.0]], "lat": [[2.0, 4.0]], "ele": [[5.0, 6.0]]})

    assert_frame_equal(
        df.select(geometry.multipoint("lon", "lat", z="ele")),
        df.select(geometry.multipoint_from_columns("lon", "lat", z="ele")),
    )
