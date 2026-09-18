"""Behaviour of the `geoarrow.linestring` extension dtype."""

from __future__ import annotations

import polars as pl
import pytest
from geopolars.datatypes import GeoLineString
from polars.exceptions import SchemaError
from polars.testing import assert_frame_equal

from tests.unit.conftest import XY, XYZ, Dimension, line_coordinates


def test_dtype_survives_a_lazy_round_trip(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A collect rebuilds the dtype from what crosses the Rust boundary,
    so the dimension has to be recoverable from the column alone."""
    lf = (
        line_coords.lazy()
        .group_by("line", maintain_order=True)
        .agg(dimension.point())
        .select(dimension.linestring())
    )
    df = lf.collect()

    assert lf.collect_schema()["line"] == dimension.linestring_dtype()
    assert df.schema["line"] == dimension.linestring_dtype()
    assert_frame_equal(line_coordinates(df), line_coords.select(dimension.coords))


def test_dtype_renders_its_dimension(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A frame header shows which coordinates the vertices carry.
    (otherwise all four dimensions would be indistinguishable on screen)"""
    df = dimension.lines(line_coords)
    tag = "".join(dimension.coords)

    assert f"linestring[{tag}]" in str(df)
    assert repr(df.schema["line"]) == dimension.linestring_dtype.__name__


def test_linestrings_of_different_dimensions_do_not_stack(
    line_coords: pl.DataFrame,
) -> None:
    with pytest.raises(SchemaError):
        pl.concat([XY.lines(line_coords), XYZ.lines(line_coords)])


def test_linestrings_of_the_same_dimension_stack(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Matching dimensions must be able to concatenate,
    and stay a linestring column while doing it."""
    df = dimension.lines(line_coords)
    stacked = pl.concat([df, df])

    assert stacked.schema["line"] == dimension.linestring_dtype()
    assert stacked.height == 2 * df.height


def test_a_linestring_is_not_a_point(
    coords: pl.DataFrame, line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Two extension names, so two dtypes, whatever the dimension."""
    points = coords.select(dimension.point())
    lines = dimension.lines(line_coords)

    assert lines.schema["line"] != points.schema["point"]
    with pytest.raises(SchemaError):
        pl.concat([lines, points.rename({"point": "line"})], how="vertical")


def test_storage_is_a_list_of_coordinate_structs(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A linestring stores `List<Coordinate>`, per the spec, and individual
    coordinates must stay accessible."""
    df = dimension.lines(line_coords)
    storage = df.select(pl.col("line").ext.storage())

    assert storage.schema["line"] == pl.List(
        pl.Struct(dict.fromkeys(dimension.coords, pl.Float64))
    )
    assert_frame_equal(line_coordinates(df), line_coords.select(dimension.coords))


def test_the_list_keeps_the_lines_apart(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The offsets are the whole point of the extra nesting: five vertices are
    two linestrings, of three and of two."""
    df = dimension.lines(line_coords)

    assert df.height == 2
    assert df.select(pl.col("line").ext.storage().list.len())["line"].to_list() == [
        3,
        2,
    ]


@pytest.mark.parametrize(
    "storage",
    [
        # A linestring nests its coordinates; a bare coordinate is a point.
        pl.Struct({"x": pl.Float64, "y": pl.Float64}),
        # Order is significant inside the coordinate, one level down.
        pl.List(pl.Struct({"y": pl.Float64, "x": pl.Float64})),
        # Interleaved coordinates are a layout this version does not implement.
        pl.List(pl.Float64),
        # One list too many: that nesting is a polygon, not a linestring.
        pl.List(pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))),
    ],
    ids=["unnested", "reversed", "interleaved", "over-nested"],
)
def test_rejects_storage_that_is_not_a_list_of_spec_coordinates(
    storage: pl.DataType,
) -> None:
    with pytest.raises(ValueError, match="unsupported 'geoarrow.linestring' storage"):
        GeoLineString.ext_from_params("geoarrow.linestring", storage, None)
