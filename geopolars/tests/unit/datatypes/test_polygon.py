"""Behaviour of the `geoarrow.polygon` extension dtype."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import SchemaError
from polars.testing import assert_frame_equal

from geopolars.datatypes import GeoPolygon
from tests.unit.conftest import XY, XYZ, Dimension, ring_coordinates


def test_dtype_survives_a_lazy_round_trip(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A collect rebuilds the dtype from what crosses the Rust boundary,
    so the dimension has to be recoverable from the column alone."""
    lf = (
        ring_coords.lazy()
        .group_by("polygon", "ring", maintain_order=True)
        .agg(dimension.point())
        .select("polygon", dimension.linestring())
        .group_by("polygon", maintain_order=True)
        .agg("line")
        .select(dimension.polygon())
    )
    df = lf.collect()

    assert lf.collect_schema()["polygon"] == dimension.polygon_dtype()
    assert df.schema["polygon"] == dimension.polygon_dtype()
    assert_frame_equal(ring_coordinates(df), ring_coords.select(dimension.coords))


def test_dtype_renders_its_dimension(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A frame header shows which coordinates the vertices carry.
    (otherwise all four dimensions would be indistinguishable on screen)"""
    df = dimension.polygons(ring_coords)
    tag = "".join(dimension.coords)

    assert f"polygon[{tag}]" in str(df)
    assert repr(df.schema["polygon"]) == dimension.polygon_dtype.__name__


def test_polygons_of_different_dimensions_do_not_stack(
    ring_coords: pl.DataFrame,
) -> None:
    with pytest.raises(SchemaError):
        pl.concat([XY.polygons(ring_coords), XYZ.polygons(ring_coords)])


def test_polygons_of_the_same_dimension_stack(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Matching dimensions must be able to concatenate,
    and stay a polygon column while doing it."""
    df = dimension.polygons(ring_coords)
    stacked = pl.concat([df, df])

    assert stacked.schema["polygon"] == dimension.polygon_dtype()
    assert stacked.height == 2 * df.height


def test_a_polygon_is_not_a_linestring(
    ring_coords: pl.DataFrame, line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A ring is a linestring, but a polygon is not: two extension names,
    so two dtypes, whatever the dimension."""
    polygons = dimension.polygons(ring_coords)
    lines = dimension.lines(line_coords)

    assert polygons.schema["polygon"] != lines.schema["line"]
    with pytest.raises(SchemaError):
        pl.concat([polygons, lines.rename({"line": "polygon"})], how="vertical")


def test_storage_is_a_list_of_lists_of_coordinate_structs(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon stores `List<List<Coordinate>>`, per the spec, and individual
    coordinates must stay accessible."""
    df = dimension.polygons(ring_coords)
    storage = df.select(pl.col("polygon").ext.storage())

    assert storage.schema["polygon"] == pl.List(
        pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
    )
    assert_frame_equal(ring_coordinates(df), ring_coords.select(dimension.coords))


def test_the_lists_keep_the_rings_apart(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The extra nesting is what separates a polygon's rings from its vertices:
    thirteen vertices are three rings, which are two polygons."""
    df = dimension.polygons(ring_coords)
    rings = pl.col("polygon").ext.storage()

    assert df.height == 2
    assert df.select(rings.list.len())["polygon"].to_list() == [2, 1]
    assert df.select(rings.explode(empty_as_null=False).list.len())[
        "polygon"
    ].to_list() == [5, 4, 4]


def test_the_first_ring_is_the_exterior_one(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Ring order carries meaning:
    the first ring of a polygon is its boundary,
    the ones after it are holes.
    So the rings arrive in the order they were given,
    not sorted or otherwise rearranged."""
    df = dimension.polygons(ring_coords)
    exterior = df.select(
        pl.col("polygon").ext.storage().list.first().alias("line")
    ).head(1)
    expected = ring_coords.filter(polygon="a", ring=0)

    assert_frame_equal(
        exterior.select(pl.col("line").explode(empty_as_null=False))
        .to_series()
        .struct.unnest(),
        expected.select(dimension.coords),
    )


@pytest.mark.parametrize(
    "storage",
    [
        # A polygon nests twice; a bare coordinate is a point.
        pl.Struct({"x": pl.Float64, "y": pl.Float64}),
        # One list short: that nesting is a linestring, a single ring.
        pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64})),
        # Order is significant inside the coordinate, two levels down.
        pl.List(pl.List(pl.Struct({"y": pl.Float64, "x": pl.Float64}))),
        # Interleaved coordinates are a layout this version does not implement.
        pl.List(pl.List(pl.Float64)),
        # One list too many: that nesting is a multipolygon.
        pl.List(pl.List(pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64})))),
    ],
    ids=["unnested", "one ring", "reversed", "interleaved", "over-nested"],
)
def test_rejects_storage_that_is_not_two_lists_of_spec_coordinates(
    storage: pl.DataType,
) -> None:
    with pytest.raises(ValueError, match="unsupported 'geoarrow.polygon' storage"):
        GeoPolygon.ext_from_params("geoarrow.polygon", storage, None)
