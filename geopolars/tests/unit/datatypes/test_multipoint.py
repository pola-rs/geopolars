"""Behaviour of the `geoarrow.multipoint` extension dtype."""


# A multipoint shares its storage with a linestring,
# so a lot of focus here is on making sure that the two stay separate.

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import SchemaError
from polars.testing import assert_frame_equal

from geopolars.datatypes import GeoMultiPoint, MultiPointXY
from tests.unit.conftest import XY, XYZ, Dimension, multipoint_coordinates


def test_dtype_survives_a_lazy_round_trip(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A collect rebuilds the dtype from what crosses the Rust boundary,
    so the dimension has to be recoverable from the column alone."""
    lf = (
        line_coords.lazy()
        .group_by("line", maintain_order=True)
        .agg(dimension.point())
        .select(dimension.multipoint())
    )
    df = lf.collect()

    assert lf.collect_schema()["multipoint"] == dimension.multipoint_dtype()
    assert df.schema["multipoint"] == dimension.multipoint_dtype()
    assert_frame_equal(multipoint_coordinates(df), line_coords.select(dimension.coords))


def test_dtype_renders_its_dimension(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A frame header shows which coordinates the points carry."""
    df = dimension.multipoints(line_coords)
    tag = "".join(dimension.coords)

    assert f"multipoint[{tag}]" in str(df)
    assert repr(df.schema["multipoint"]) == dimension.multipoint_dtype.__name__


def test_multipoints_of_different_dimensions_do_not_stack(
    line_coords: pl.DataFrame,
) -> None:
    with pytest.raises(SchemaError):
        pl.concat([XY.multipoints(line_coords), XYZ.multipoints(line_coords)])


def test_multipoints_of_the_same_dimension_stack(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)
    stacked = pl.concat([df, df])

    assert stacked.schema["multipoint"] == dimension.multipoint_dtype()
    assert stacked.height == 2 * df.height


def test_a_multipoint_is_not_a_point(
    coords: pl.DataFrame, line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    points = coords.select(dimension.point())
    multipoints = dimension.multipoints(line_coords)

    assert multipoints.schema["multipoint"] != points.schema["point"]
    with pytest.raises(SchemaError):
        pl.concat([multipoints, points.rename({"point": "multipoint"})], how="vertical")


def test_a_multipoint_is_not_a_linestring(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """These two have byte-identical storage,
    but the spec still calls them different geometries."""
    multipoints = dimension.multipoints(line_coords)
    lines = dimension.lines(line_coords)

    assert (
        multipoints.schema["multipoint"].ext_storage()
        == lines.schema["line"].ext_storage()
    )
    assert multipoints.schema["multipoint"] != lines.schema["line"]
    with pytest.raises(SchemaError):
        pl.concat([multipoints, lines.rename({"line": "multipoint"})], how="vertical")


def test_storage_is_a_list_of_coordinate_structs(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)
    storage = df.select(pl.col("multipoint").ext.storage())

    assert storage.schema["multipoint"] == pl.List(
        pl.Struct(dict.fromkeys(dimension.coords, pl.Float64))
    )
    assert_frame_equal(multipoint_coordinates(df), line_coords.select(dimension.coords))


def test_the_list_keeps_the_multipoints_apart(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The five points are two multipoints, of three and of two."""
    df = dimension.multipoints(line_coords)

    assert df.height == 2
    assert df.select(pl.col("multipoint").ext.storage().list.len())[
        "multipoint"
    ].to_list() == [3, 2]


@pytest.mark.parametrize(
    "storage",
    [
        # A multipoint nests its coordinates; a bare coordinate is a point.
        pl.Struct({"x": pl.Float64, "y": pl.Float64}),
        # Order is significant inside the coordinate, one level down.
        pl.List(pl.Struct({"y": pl.Float64, "x": pl.Float64})),
        # Interleaved coordinates are a layout this version does not implement.
        pl.List(pl.Float64),
        # One list too many: that nesting is a multilinestring, not a multipoint.
        pl.List(pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))),
    ],
    ids=["unnested", "reversed", "interleaved", "over-nested"],
)
def test_rejects_storage_that_is_not_a_list_of_spec_coordinates(
    storage: pl.DataType,
) -> None:
    with pytest.raises(ValueError, match="unsupported 'geoarrow.multipoint' storage"):
        GeoMultiPoint.ext_from_params("geoarrow.multipoint", storage, None)


def test_the_extension_name_is_the_one_the_spec_fixes() -> None:
    assert MultiPointXY().ext_name() == "geoarrow.multipoint"
    assert GeoMultiPoint._extension_name == "geoarrow.multipoint"
