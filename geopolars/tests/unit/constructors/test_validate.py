"""Testing validation function:
GeoArrow allows nulls only at the outermost level.
<https://geoarrow.org/format.html#missing-values-null>
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geometry
from geopolars.datatypes import LineStringXY, PointXY, PolygonXY

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_XY_VERTICES = pl.List(_XY)
_XY_RINGS = pl.List(_XY_VERTICES)

_SQUARE = [
    {"x": 0.0, "y": 0.0},
    {"x": 4.0, "y": 0.0},
    {"x": 4.0, "y": 4.0},
    {"x": 0.0, "y": 0.0},
]


def test_a_missing_vertex_nulls_the_linestring_around_it() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, None], _SQUARE]},
        schema={"line": _XY_VERTICES},
    ).select(pl.col("line").ext.to(LineStringXY()))

    out = df.select(geometry.validate("line"))

    assert out["line"].is_null().to_list() == [True, False]


def test_a_missing_coordinate_nulls_the_linestring_around_it() -> None:
    """A vertex that is there but missing an axis is a vertex that is not
    there: the geometry cannot stand without it either."""
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": None}]]},
        schema={"line": _XY_VERTICES},
    ).select(pl.col("line").ext.to(LineStringXY()))

    out = df.select(geometry.validate("line"))

    assert out["line"].is_null().to_list() == [True]


def test_a_missing_ring_nulls_the_polygon_around_it() -> None:
    """Nesting is walked all the way down: a null one layer in is as fatal as
    a null coordinate two layers in."""
    df = pl.DataFrame(
        {"polygon": [[_SQUARE, None], [_SQUARE]]}, schema={"polygon": _XY_RINGS}
    ).select(pl.col("polygon").ext.to(PolygonXY()))

    out = df.select(geometry.validate("polygon"))

    assert out["polygon"].is_null().to_list() == [True, False]


def test_a_missing_coordinate_inside_a_ring_nulls_the_polygon() -> None:
    df = pl.DataFrame(
        {"polygon": [[[{"x": 0.0, "y": None}]]]}, schema={"polygon": _XY_RINGS}
    ).select(pl.col("polygon").ext.to(PolygonXY()))

    out = df.select(geometry.validate("polygon"))

    assert out["polygon"].is_null().to_list() == [True]


def test_a_missing_coordinate_nulls_the_point() -> None:
    df = pl.DataFrame(
        {"point": [{"x": 1.0, "y": None}, {"x": 1.0, "y": 2.0}]},
        schema={"point": _XY},
    ).select(pl.col("point").ext.to(PointXY()))

    out = df.select(geometry.validate("point"))

    assert out["point"].is_null().to_list() == [True, False]


def test_an_empty_geometry_is_whole() -> None:
    """A geometry with no coordinates is not a geometry with missing ones. It
    has no centroid, but it is there, and `validate` leaves it alone."""
    df = pl.DataFrame({"line": [[], _SQUARE]}, schema={"line": _XY_VERTICES}).select(
        pl.col("line").ext.to(LineStringXY())
    )

    out = df.select(geometry.validate("line"))

    assert out["line"].is_null().to_list() == [False, False]


def test_a_null_geometry_stays_null() -> None:
    df = pl.DataFrame({"line": [None, _SQUARE]}, schema={"line": _XY_VERTICES}).select(
        pl.col("line").ext.to(LineStringXY())
    )

    out = df.select(geometry.validate("line"))

    assert out["line"].is_null().to_list() == [True, False]


def test_it_leaves_a_whole_column_exactly_as_it_was() -> None:
    df = pl.DataFrame({"line": [_SQUARE, []]}, schema={"line": _XY_VERTICES}).select(
        geometry.linestring("line")
    )

    assert_frame_equal(df.select(geometry.validate("line")), df)


def test_it_keeps_the_dtype_it_was_given() -> None:
    df = pl.DataFrame({"line": [_SQUARE]}, schema={"line": _XY_VERTICES}).select(
        pl.col("line").ext.to(LineStringXY())
    )

    out = df.select(geometry.validate("line"))

    assert out.schema["line"] == LineStringXY()


def test_it_is_idempotent() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, None], _SQUARE]},
        schema={"line": _XY_VERTICES},
    ).select(pl.col("line").ext.to(LineStringXY()))
    once = df.select(geometry.validate("line"))

    assert_frame_equal(once.select(geometry.validate("line")), once)


def test_namespace_matches_the_functional_api() -> None:
    df = pl.DataFrame(
        {"line": [[{"x": 0.0, "y": 0.0}, None]]}, schema={"line": _XY_VERTICES}
    ).select(pl.col("line").ext.to(LineStringXY()))

    assert_frame_equal(
        df.select(gpl.col("line").geometry.validate()),
        df.select(geometry.validate("line")),
    )


def test_it_runs_on_the_streaming_engine() -> None:
    lf = (
        pl.LazyFrame(
            {"line": [[{"x": 0.0, "y": 0.0}, None]]}, schema={"line": _XY_VERTICES}
        )
        .select(pl.col("line").ext.to(LineStringXY()))
        .select(geometry.validate("line"))
    )

    assert_frame_equal(lf.collect(engine="streaming"), lf.collect())


def test_rejects_a_column_that_is_not_a_geometry() -> None:
    df = pl.DataFrame({"line": [1.0]})

    with pytest.raises(ComputeError, match="expected a `geoarrow.point`"):
        df.select(geometry.validate("line"))


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    """The output type is read off the input's dtype, so a bad column is a
    schema error and not something that waits until the data is there."""
    lf = pl.LazyFrame({"line": [1.0]}).select(geometry.validate("line"))

    with pytest.raises(ComputeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()
