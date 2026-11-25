"""Testing `null_out_invalid_geoarrow`:
GeoArrow allows nulls only at the outermost level.
<https://geoarrow.org/format.html#missing-values-null>
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import BoxXY, LineStringXY, PointXY, PolygonXY

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_XY_VERTICES = pl.List(_XY)
_XY_RINGS = pl.List(_XY_VERTICES)

_SQUARE = [
    {"x": 0.0, "y": 0.0},
    {"x": 4.0, "y": 0.0},
    {"x": 4.0, "y": 4.0},
    {"x": 0.0, "y": 0.0},
]


def test_a_null_vertex_nulls_the_linestring_around_it() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, None], _SQUARE]},
        schema={"line": _XY_VERTICES},
    ).select(pl.col("line").ext.to(LineStringXY()))

    out = df.select(geo.null_out_invalid_geoarrow("line"))

    assert out["line"].is_null().to_list() == [True, False]


def test_a_null_coordinate_nulls_the_linestring_around_it() -> None:
    """A vertex that is there but has a null axis is a vertex that is not
    there: the geometry cannot stand without it either."""
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": None}]]},
        schema={"line": _XY_VERTICES},
    ).select(pl.col("line").ext.to(LineStringXY()))

    out = df.select(geo.null_out_invalid_geoarrow("line"))

    assert out["line"].is_null().to_list() == [True]


def test_a_null_ring_nulls_the_polygon_around_it() -> None:
    """Nesting is walked all the way down: a null one layer in is as fatal as
    a null coordinate two layers in."""
    df = pl.DataFrame(
        {"polygon": [[_SQUARE, None], [_SQUARE]]}, schema={"polygon": _XY_RINGS}
    ).select(pl.col("polygon").ext.to(PolygonXY()))

    out = df.select(geo.null_out_invalid_geoarrow("polygon"))

    assert out["polygon"].is_null().to_list() == [True, False]


def test_a_null_coordinate_inside_a_ring_nulls_the_polygon() -> None:
    df = pl.DataFrame(
        {"polygon": [[[{"x": 0.0, "y": None}]]]}, schema={"polygon": _XY_RINGS}
    ).select(pl.col("polygon").ext.to(PolygonXY()))

    out = df.select(geo.null_out_invalid_geoarrow("polygon"))

    assert out["polygon"].is_null().to_list() == [True]


def test_a_null_coordinate_nulls_the_point() -> None:
    df = pl.DataFrame(
        {"point": [{"x": 1.0, "y": None}, {"x": 1.0, "y": 2.0}]},
        schema={"point": _XY},
    ).select(pl.col("point").ext.to(PointXY()))

    out = df.select(geo.null_out_invalid_geoarrow("point"))

    assert out["point"].is_null().to_list() == [True, False]


def test_a_nan_coordinate_nulls_the_linestring_around_it() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, {"x": math.nan, "y": 1.0}], _SQUARE]},
        schema={"line": _XY_VERTICES},
    ).select(pl.col("line").ext.to(LineStringXY()))

    out = df.select(
        geo.null_out_invalid_geoarrow("line", check_finite_coordinates=True)
    )

    assert out["line"].is_null().to_list() == [True, False]


def test_an_open_ring_nulls_the_polygon() -> None:
    df = pl.DataFrame(
        {"polygon": [[_SQUARE[:-1]], [_SQUARE]]}, schema={"polygon": _XY_RINGS}
    ).select(pl.col("polygon").ext.to(PolygonXY()))

    out = df.select(geo.null_out_invalid_geoarrow("polygon"))

    assert out["polygon"].is_null().to_list() == [True, False]


def test_a_point_outside_its_crs_is_nulled_and_the_crs_kept() -> None:
    df = pl.DataFrame(
        {"point": [{"x": 0.0, "y": 95.0}, {"x": 190.0, "y": 0.0}]},
        schema={"point": _XY},
    ).select(pl.col("point").ext.to(PointXY(crs="EPSG:4326")))

    out = df.select(
        geo.null_out_invalid_geoarrow("point", out_of_bounds=geo.OutOfBounds())
    )

    assert out["point"].is_null().to_list() == [True, True]
    assert out.schema == df.schema


_BOX = {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0}


def test_a_box_without_any_bound_is_nulled_and_the_crs_kept() -> None:
    """The way geopandas writes the box of a null geometry."""
    df = pl.DataFrame(
        {"bbox": [_BOX, dict.fromkeys(_BOX), None]},
        schema={"bbox": BoxXY().ext_storage()},
    ).select(pl.col("bbox").ext.to(BoxXY(crs="EPSG:4326")))

    out = df.select(geo.null_out_invalid_geoarrow("bbox"))

    assert out["bbox"].ext.storage().to_list() == [_BOX, None, None]
    assert out.schema == df.schema


def test_a_non_finite_box_is_nulled() -> None:
    empty = _BOX | {"xmin": math.inf, "xmax": -math.inf}
    df = pl.DataFrame(
        {"bbox": [_BOX | {"ymin": math.nan}, empty]},
        schema={"bbox": BoxXY().ext_storage()},
    ).select(pl.col("bbox").ext.to(BoxXY()))

    out = df.select(
        geo.null_out_invalid_geoarrow("bbox", check_finite_coordinates=True)
    )

    assert out["bbox"].is_null().to_list() == [True, False]


def test_it_leaves_a_whole_column_of_boxes_exactly_as_it_was() -> None:
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [2.0, 3.0]}).select(
        geo.box("x", "y", "y", "y").alias("bbox")
    )

    assert_frame_equal(df.select(geo.null_out_invalid_geoarrow("bbox")), df)


def test_every_chunk_is_checked() -> None:
    chunk = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, None], _SQUARE]},
        schema={"line": _XY_VERTICES},
    ).select(pl.col("line").ext.to(LineStringXY()))
    df = pl.concat([chunk, chunk], rechunk=False)

    out = df.select(geo.null_out_invalid_geoarrow("line"))

    assert out["line"].is_null().to_list() == [True, False, True, False]


def test_an_empty_geometry_is_whole() -> None:
    """A geometry with no coordinates is not a geometry with null ones. It
    has no mean coordinate, but it is there, and `null_out_invalid_geoarrow` leaves it alone."""
    df = pl.DataFrame({"line": [[], _SQUARE]}, schema={"line": _XY_VERTICES}).select(
        pl.col("line").ext.to(LineStringXY())
    )

    out = df.select(geo.null_out_invalid_geoarrow("line"))

    assert out["line"].is_null().to_list() == [False, False]


def test_a_null_geometry_stays_null() -> None:
    df = pl.DataFrame({"line": [None, _SQUARE]}, schema={"line": _XY_VERTICES}).select(
        pl.col("line").ext.to(LineStringXY())
    )

    out = df.select(geo.null_out_invalid_geoarrow("line"))

    assert out["line"].is_null().to_list() == [True, False]


def test_it_leaves_a_whole_column_exactly_as_it_was() -> None:
    df = pl.DataFrame({"line": [_SQUARE, []]}, schema={"line": _XY_VERTICES}).select(
        geo.line_string("line")
    )

    assert_frame_equal(df.select(geo.null_out_invalid_geoarrow("line")), df)


def test_it_is_idempotent() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, None], _SQUARE]},
        schema={"line": _XY_VERTICES},
    ).select(pl.col("line").ext.to(LineStringXY()))
    once = df.select(geo.null_out_invalid_geoarrow("line"))

    assert_frame_equal(once.select(geo.null_out_invalid_geoarrow("line")), once)


def test_namespace_matches_the_functional_api() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, None]]}, schema={"line": _XY_VERTICES}
    ).select(pl.col("line").ext.to(LineStringXY()))

    assert_frame_equal(
        df.select(gpl.col("line").geo.null_out_invalid_geoarrow()),
        df.select(geo.null_out_invalid_geoarrow("line")),
    )


def test_it_runs_on_the_streaming_engine() -> None:
    lf = (
        pl.LazyFrame(
            {"line": [[{"x": 0.0, "y": 0.0}, None]]}, schema={"line": _XY_VERTICES}
        )
        .select(pl.col("line").ext.to(LineStringXY()))
        .select(geo.null_out_invalid_geoarrow("line"))
    )

    assert_frame_equal(lf.collect(engine="streaming"), lf.collect())


def test_rejects_a_column_that_is_not_a_geometry() -> None:
    df = pl.DataFrame({"line": [1.0]})

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        df.select(geo.null_out_invalid_geoarrow("line"))


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    """The dtype is checked while the plan is built,
    not when the data is there."""
    lf = pl.LazyFrame({"line": [1.0]}).select(geo.null_out_invalid_geoarrow("line"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()
