"""Unit tests for the `geometry` namespace"""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geometry
from tests.unit.conftest import (
    XY,
    XYM,
    XYZ,
    XYZM,
    Dimension,
    coordinates,
    line_coordinates,
    multipoint_coordinates,
    ring_coordinates,
)


def test_translate_shifts_x_and_y(coords: pl.DataFrame, dimension: Dimension) -> None:
    df = coords.select(dimension.point())
    out = df.select(geometry.translate("point", dx=1.5, dy=-2.0).alias("point"))

    assert_frame_equal(
        coordinates(out),
        coordinates(df).with_columns(pl.col("x") + 1.5, pl.col("y") - 2.0),
    )


def test_translate_keeps_the_dimension(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Moving a point does not change what kind of point it is."""
    df = coords.select(dimension.point())
    out = df.select(geometry.translate("point", dx=1.0, dy=1.0).alias("point"))

    assert out.schema["point"] == dimension.point_dtype()


def test_translate_by_zero_is_the_identity(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())
    out = df.select(geometry.translate("point", dx=0.0, dy=0.0).alias("point"))

    assert_frame_equal(out, df)


def test_translate_shifts_z(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point())
    out = df.select(geometry.translate("point", dx=0.0, dy=0.0, dz=5.0).alias("point"))

    assert_frame_equal(
        coordinates(out), coordinates(df).with_columns(pl.col("z") + 5.0)
    )


def test_translate_leaves_m_untouched(coords: pl.DataFrame) -> None:
    """`m` is a measure, not a position: moving the geometry must not change
    the timestamp or distance-along-route the vertex carries."""
    df = coords.select(XYZM.point())
    out = df.select(
        geometry.translate("point", dx=10.0, dy=10.0, dz=10.0).alias("point")
    )

    assert_frame_equal(coordinates(out).select("m"), coordinates(df).select("m"))


@pytest.mark.parametrize(
    "dimension",
    [
        XY,
        XYM,
    ],
    ids=["PointXY", "PointXYM"],
)
def test_translate_rejects_dz_on_a_point_with_no_z(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())

    with pytest.raises(ComputeError, match="cannot translate by dz"):
        df.select(geometry.translate("point", dx=0.0, dy=0.0, dz=1.0))


@pytest.mark.parametrize(
    "dimension",
    [
        XY,
        XYM,
    ],
    ids=["PointXY", "PointXYM"],
)
def test_translate_allows_a_zero_dz_on_a_point_with_no_z(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """dz defaults to 0.0, so an ordinary 2D translate must not trip the check."""
    df = coords.select(dimension.point())
    out = df.select(geometry.translate("point", dx=1.0, dy=1.0, dz=0.0).alias("point"))

    assert out.schema["point"] == dimension.point_dtype()


def test_translate_rejects_a_plain_float_column() -> None:
    df = pl.DataFrame({"point": [1.0, 2.0]})

    with pytest.raises(ComputeError, match="expected a `geoarrow.point`"):
        df.select(geometry.translate("point", dx=1.0, dy=1.0))


def test_translate_rejects_a_bare_coordinate_struct() -> None:
    df = pl.DataFrame({"point": [{"x": 1.0, "y": 2.0}]})

    with pytest.raises(ComputeError, match="expected a `geoarrow.point`"):
        df.select(geometry.translate("point", dx=1.0, dy=1.0))


def test_translate_keeps_a_missing_point_missing() -> None:
    """Shifting the coordinates rebuilds the storage struct, and the row that
    says 'no point here' must survive that: GeoArrow has no point with null
    coordinates, only a null point."""
    df = pl.DataFrame({"x": [None, 1.0], "y": [2.0, 2.0]}).select(
        geometry.point("x", "y").alias("point")
    )
    out = df.select(geometry.translate("point", dx=1.0, dy=1.0).alias("point"))

    assert out["point"].is_null().to_list() == [True, False]


def test_translate_shifts_every_vertex(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A linestring moves as a whole: the same offset applies to each of its
    vertices, and which vertices belong to which line does not change."""
    df = dimension.lines(line_coords)
    out = df.select(geometry.translate("line", dx=1.5, dy=-2.0).alias("line"))

    assert out.schema["line"] == dimension.linestring_dtype()
    assert_frame_equal(
        line_coordinates(out),
        line_coordinates(df).with_columns(pl.col("x") + 1.5, pl.col("y") - 2.0),
    )
    assert_frame_equal(
        out.select(pl.col("line").ext.storage().list.len()),
        df.select(pl.col("line").ext.storage().list.len()),
    )


def test_translate_leaves_a_linestrings_measures_untouched(
    line_coords: pl.DataFrame,
) -> None:
    df = XYZM.lines(line_coords)
    out = df.select(geometry.translate("line", dx=10.0, dy=10.0, dz=10.0).alias("line"))

    assert_frame_equal(
        line_coordinates(out).select("m"), line_coordinates(df).select("m")
    )


def test_translate_rejects_dz_on_a_linestring_with_no_z(
    line_coords: pl.DataFrame,
) -> None:
    df = XY.lines(line_coords)

    with pytest.raises(ComputeError, match="cannot translate by dz"):
        df.select(geometry.translate("line", dx=0.0, dy=0.0, dz=1.0))


def test_translate_keeps_empty_and_missing_linestrings(dimension: Dimension) -> None:
    """An empty line has nothing to shift and a missing one is still missing."""
    df = pl.DataFrame(
        {"vertices": [[], None]},
        schema={
            "vertices": pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
        },
    ).select(geometry.linestring("vertices").alias("line"))
    out = df.select(geometry.translate("line", dx=1.0, dy=1.0).alias("line"))

    assert_frame_equal(out, df)


def test_translate_shifts_every_point_of_a_multipoint(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)
    out = df.select(
        geometry.translate("multipoint", dx=1.5, dy=-2.0).alias("multipoint")
    )

    assert out.schema["multipoint"] == dimension.multipoint_dtype()
    assert_frame_equal(
        multipoint_coordinates(out),
        multipoint_coordinates(df).with_columns(pl.col("x") + 1.5, pl.col("y") - 2.0),
    )
    assert_frame_equal(
        out.select(pl.col("multipoint").ext.storage().list.len()),
        df.select(pl.col("multipoint").ext.storage().list.len()),
    )


def test_translate_leaves_a_multipoints_measures_untouched(
    line_coords: pl.DataFrame,
) -> None:
    df = XYZM.multipoints(line_coords)
    out = df.select(
        geometry.translate("multipoint", dx=10.0, dy=10.0, dz=10.0).alias("multipoint")
    )

    assert_frame_equal(
        multipoint_coordinates(out).select("m"),
        multipoint_coordinates(df).select("m"),
    )


def test_translate_rejects_dz_on_a_multipoint_with_no_z(
    line_coords: pl.DataFrame,
) -> None:
    df = XY.multipoints(line_coords)

    with pytest.raises(ComputeError, match="cannot translate by dz"):
        df.select(geometry.translate("multipoint", dx=0.0, dy=0.0, dz=1.0))


def test_translate_shifts_every_vertex_of_every_ring(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon moves as a whole."""
    df = dimension.polygons(ring_coords)
    out = df.select(geometry.translate("polygon", dx=1.5, dy=-2.0).alias("polygon"))
    rings = pl.col("polygon").ext.storage()

    assert out.schema["polygon"] == dimension.polygon_dtype()
    assert_frame_equal(
        ring_coordinates(out),
        ring_coordinates(df).with_columns(pl.col("x") + 1.5, pl.col("y") - 2.0),
    )
    assert_frame_equal(out.select(rings.list.len()), df.select(rings.list.len()))
    assert_frame_equal(
        out.select(rings.explode(empty_as_null=False).list.len()),
        df.select(rings.explode(empty_as_null=False).list.len()),
    )


def test_translate_keeps_empty_and_missing_polygons(dimension: Dimension) -> None:
    df = pl.DataFrame(
        {"rings": [[], [[]], None]},
        schema={
            "rings": pl.List(
                pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
            )
        },
    ).select(geometry.polygon("rings").alias("polygon"))
    out = df.select(geometry.translate("polygon", dx=1.0, dy=1.0).alias("polygon"))

    assert_frame_equal(out, df)


def test_namespace_matches_the_functional_api(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point())

    assert_frame_equal(
        df.select(gpl.col("point").geometry.translate(1.0, 2.0, 3.0).alias("point")),
        df.select(geometry.translate("point", dx=1.0, dy=2.0, dz=3.0).alias("point")),
    )


def test_linestring_namespace_matches_the_functional_api(
    line_coords: pl.DataFrame,
) -> None:
    vertices = line_coords.group_by("line", maintain_order=True).agg(XYZ.point())

    assert_frame_equal(
        vertices.select(gpl.col("point").geometry.linestring().alias("line")),
        vertices.select(geometry.linestring("point").alias("line")),
    )


def test_multipoint_namespace_matches_the_functional_api(
    line_coords: pl.DataFrame,
) -> None:
    points = line_coords.group_by("line", maintain_order=True).agg(XYZ.point())

    assert_frame_equal(
        points.select(gpl.col("point").geometry.multipoint().alias("multipoint")),
        points.select(geometry.multipoint("point").alias("multipoint")),
    )


def test_polygon_namespace_matches_the_functional_api(
    ring_coords: pl.DataFrame,
) -> None:
    rings = (
        ring_coords.group_by("polygon", "ring", maintain_order=True)
        .agg(XYZ.point())
        .select("polygon", XYZ.linestring())
        .group_by("polygon", maintain_order=True)
        .agg("line")
    )

    assert_frame_equal(
        rings.select(gpl.col("line").geometry.polygon().alias("polygon")),
        rings.select(geometry.polygon("line").alias("polygon")),
    )


def test_translate_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    """The output type is derived from the input's dtype,
    so a bad column is a schema error."""
    lf = pl.LazyFrame({"lon": [1.0]}).select(geometry.translate("lon", dx=1.0, dy=1.0))

    with pytest.raises(ComputeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()
