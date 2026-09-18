"""The centroid of the coordinates a geometry is made of."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError, StructFieldNotFoundError
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geometry
from geopolars.datatypes import (
    GeoPoint,
    LineStringXY,
    PointXY,
    PolygonXY,
)
from tests.unit.conftest import XY, XYZM, Dimension, coordinates

_SQUARE = [
    {"x": 0.0, "y": 0.0},
    {"x": 4.0, "y": 0.0},
    {"x": 4.0, "y": 4.0},
    {"x": 0.0, "y": 4.0},
    {"x": 0.0, "y": 0.0},
]

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)


def _mean_per(vertices: pl.DataFrame, group: str, dimension: Dimension) -> pl.DataFrame:
    """The mean of each coordinate, per geometry, straight off a vertex frame."""
    return (
        vertices.group_by(group, maintain_order=True)
        .agg(pl.col(*dimension.coords).mean())
        .select(*dimension.coords)
    )


def _opened(rings: pl.DataFrame) -> pl.DataFrame:
    """A vertex frame with the closing vertex of every ring dropped."""
    within = pl.int_range(pl.len()).over("polygon", "ring")
    return rings.filter(within < pl.len().over("polygon", "ring") - 1)


def test_a_point_is_its_own_centroid(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())
    out = df.select(geometry.coordinate_centroid("point"))

    assert out.schema["point"] == dimension.point_dtype()
    assert_frame_equal(coordinates(out), coordinates(df))


def test_a_linestring_averages_its_vertices(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.lines(line_coords)
    out = df.select(geometry.coordinate_centroid("line"))

    assert_frame_equal(
        coordinates(out, "line"), _mean_per(line_coords, "line", dimension)
    )


def test_a_polygon_averages_the_vertices_of_all_its_rings(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Holes count as much as the exterior ring: every coordinate counts once."""
    df = dimension.polygons(ring_coords)
    out = df.select(geometry.coordinate_centroid("polygon"))

    assert_frame_equal(
        coordinates(out, "polygon"),
        _mean_per(_opened(ring_coords), "polygon", dimension),
    )


def test_the_coordinate_that_closes_a_ring_is_left_out() -> None:
    """A square's four corners average to its middle. Counting the repeat that
    closes the ring would pull the centroid towards wherever it starts."""
    df = pl.DataFrame({"rings": [[_SQUARE]]}, schema={"rings": _XY_RINGS}).select(
        geometry.polygon("rings").alias("polygon")
    )
    out = df.select(geometry.coordinate_centroid("polygon"))

    assert_frame_equal(
        coordinates(out, "polygon"), pl.DataFrame({"x": [2.0], "y": [2.0]})
    )


def test_a_closed_linestring_keeps_every_vertex() -> None:
    """A linestring is not a ring, even when it closes: nothing says its last
    vertex is there to close it, so all five are averaged."""
    df = pl.DataFrame(
        {"vertices": [_SQUARE]}, schema={"vertices": _XY_VERTICES}
    ).select(geometry.linestring("vertices").alias("line"))
    out = df.select(geometry.coordinate_centroid("line"))

    assert_frame_equal(coordinates(out, "line"), pl.DataFrame({"x": [1.6], "y": [1.6]}))


def test_an_empty_ring_takes_nothing_with_it() -> None:
    df = pl.DataFrame(
        {"rings": [[_SQUARE, []], [[], _SQUARE]]}, schema={"rings": _XY_RINGS}
    ).select(geometry.polygon("rings").alias("polygon"))
    out = df.select(geometry.coordinate_centroid("polygon"))

    # The square's four corners, whichever side of it the empty ring is on.
    assert_frame_equal(
        coordinates(out, "polygon"),
        pl.DataFrame({"x": [2.0, 2.0], "y": [2.0, 2.0]}),
    )


def test_a_missing_ring_takes_the_whole_centroid_with_it() -> None:
    df = pl.DataFrame(
        {"polygon": [[_SQUARE, None]]}, schema={"polygon": _XY_RINGS}
    ).select(pl.col("polygon").ext.to(PolygonXY()))
    out = df.select(geometry.coordinate_centroid("polygon"))

    assert out["polygon"].is_null().to_list() == [True]


@pytest.mark.parametrize("builder", ["point", "line", "polygon"])
def test_the_result_is_a_point_of_the_same_dimension(
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    dimension: Dimension,
    builder: str,
) -> None:
    """Whatever went in, a point of the same dimension comes out."""
    if builder == "point":
        df = coords.select(dimension.point())
    elif builder == "line":
        df = dimension.lines(line_coords)
    else:
        df = dimension.polygons(ring_coords)
    name = df.columns[0]

    out = df.select(geometry.coordinate_centroid(name))

    assert out.schema[name] == GeoPoint.of_dimension(dimension.coords)()


def test_m_is_averaged_like_any_other_coordinate(line_coords: pl.DataFrame) -> None:
    """Unlike `translate`, which leaves a measure where it was, there is nothing
    to carry through here: the centroid of a trajectory carries the mean of the
    measures its vertices hold."""
    df = XYZM.lines(line_coords)
    out = df.select(geometry.coordinate_centroid("line"))

    assert_frame_equal(
        coordinates(out, "line").select("m"),
        _mean_per(line_coords, "line", XYZM).select("m"),
    )


def test_an_empty_or_missing_linestring_has_no_centroid(dimension: Dimension) -> None:
    """A geometry with no coordinates has no centroid, and a null geometry stays
    null: GeoArrow has no point with null coordinates, only a null point."""
    df = pl.DataFrame(
        {"vertices": [[], None]},
        schema={
            "vertices": pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
        },
    ).select(geometry.linestring("vertices").alias("line"))

    out = df.select(geometry.coordinate_centroid("line"))

    assert out["line"].is_null().to_list() == [True, True]


def test_an_empty_or_missing_polygon_has_no_centroid(dimension: Dimension) -> None:
    """A polygon with no rings, or with nothing but empty ones, has no
    coordinates to average either."""
    df = pl.DataFrame(
        {"rings": [[], [[]], None]},
        schema={
            "rings": pl.List(
                pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
            )
        },
    ).select(geometry.polygon("rings").alias("polygon"))

    out = df.select(geometry.coordinate_centroid("polygon"))

    assert out["polygon"].is_null().to_list() == [True, True, True]


def test_a_missing_point_has_no_centroid() -> None:
    df = pl.DataFrame({"x": [None, 1.0], "y": [2.0, 2.0]}).select(
        geometry.point("x", "y").alias("point")
    )

    out = df.select(geometry.coordinate_centroid("point"))

    assert out["point"].is_null().to_list() == [True, False]


def test_a_missing_coordinate_takes_the_whole_centroid_with_it() -> None:
    """The constructors do not let a vertex go missing, so this goes around
    them. Averaging what is left would hand back a centroid of a geometry that
    is not there."""
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": None}]]},
        schema={"line": _XY_VERTICES},
    ).select(pl.col("line").ext.to(LineStringXY()))

    out = df.select(geometry.coordinate_centroid("line"))

    assert out["line"].is_null().to_list() == [True]


def test_every_form_of_column_gives_the_same_answer(line_coords: pl.DataFrame) -> None:
    """A name, a `pl.col(...)` and a `Series` are the same column."""
    df = XY.lines(line_coords)
    expected = df.select(geometry.coordinate_centroid("line"))

    assert_frame_equal(
        df.select(geometry.coordinate_centroid(pl.col("line"))), expected
    )
    assert_frame_equal(pl.select(geometry.coordinate_centroid(df["line"])), expected)


def test_namespace_matches_the_functional_api(line_coords: pl.DataFrame) -> None:
    df = XY.lines(line_coords)

    assert_frame_equal(
        df.select(gpl.col("line").geometry.coordinate_centroid()),
        df.select(geometry.coordinate_centroid("line")),
    )


def test_it_runs_on_the_streaming_engine(line_coords: pl.DataFrame) -> None:
    """The point of building this out of ordinary expressions."""
    lf = XY.lines(line_coords).lazy().select(geometry.coordinate_centroid("line"))

    assert_frame_equal(lf.collect(engine="streaming"), lf.collect())


def test_rejects_a_plain_float_column() -> None:
    df = pl.DataFrame({"line": [1.0, 2.0]})

    with pytest.raises(StructFieldNotFoundError, match="expected a `geoarrow.point`"):
        df.select(geometry.coordinate_centroid("line"))


def test_rejects_a_bare_coordinate_struct() -> None:
    """The storage of a point is not a point: the dtype is what says it is one."""
    df = pl.DataFrame({"point": [{"x": 1.0, "y": 2.0}]})

    with pytest.raises(StructFieldNotFoundError, match="expected a `geoarrow.point`"):
        df.select(geometry.coordinate_centroid("point"))


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    """The geometry is read off the dtype, so a bad column is a schema error and
    not something that waits until the data is there."""
    lf = pl.LazyFrame({"lon": [1.0]}).select(geometry.coordinate_centroid("lon"))

    with pytest.raises(StructFieldNotFoundError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


def test_rejects_a_geometry_carrying_extension_metadata() -> None:
    """Metadata is part of the dtype, and a dtype we cannot name is one we
    cannot select on. Rejecting it beats dropping a CRS on the floor; this is
    the check to lift when CRS lands."""
    spherical = GeoPoint.ext_from_params(
        "geoarrow.point", PointXY().ext_storage(), '{"edges":"spherical"}'
    )
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(
        pl.struct(x=pl.col("x"), y=pl.col("y")).ext.to(spherical).alias("point")
    )

    with pytest.raises(StructFieldNotFoundError, match="no extension metadata"):
        df.select(geometry.coordinate_centroid("point"))


def test_rejects_an_expression_that_is_not_a_column(line_coords: pl.DataFrame) -> None:
    """A chained expression has no dtype until the plan is resolved, and by then
    the expression has to have been built."""
    translated = gpl.col("line").geometry.translate(1.0, 1.0)

    with pytest.raises(TypeError, match="expected a column name"):
        XY.lines(line_coords).select(geometry.coordinate_centroid(translated))


def test_rejects_a_series_that_is_not_a_geometry() -> None:
    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        pl.select(geometry.coordinate_centroid(pl.Series("line", [1.0])))


def test_rejects_a_column_that_is_not_there() -> None:
    df = pl.DataFrame({"a": [1.0]})

    with pytest.raises(ColumnNotFoundError, match="line"):
        df.select(geometry.coordinate_centroid("line"))
