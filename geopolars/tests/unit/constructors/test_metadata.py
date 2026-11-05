"""Setting GeoArrow metadata through a constructor's keyword arguments."""

from __future__ import annotations

import json
from collections.abc import Callable

import polars as pl
import pytest
from polars.exceptions import ComputeError

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointType,
    PointXY,
    PolygonXY,
)
from tests.unit.conftest import Dimension

WGS84 = "EPSG:4326"
PROJJSON = json.dumps({"type": "GeographicCRS", "name": "WGS 84"}, indent=2)


def _points(crs: str | None = None) -> pl.DataFrame:
    """One list of two points per row, ready to be gathered."""
    return (
        pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        .select(point=geo.point("x", "y", crs=crs))
        .select(pl.col("point").implode())
    )


def _lines(crs: str | None = None) -> pl.DataFrame:
    """One list of two closed lines per row, ready to be gathered."""
    return (
        pl.DataFrame({"x": [0.0, 1.0, 0.0, 0.0], "y": [0.0, 0.0, 1.0, 0.0]})
        .select(line=geo.line_string(geo.point("x", "y", crs=crs).implode()))
        .select(pl.col("line").repeat_by(2))
    )


def _polygons(crs: str | None = None) -> pl.DataFrame:
    """One list of two polygons per row, ready to be gathered."""
    return (
        _lines(crs)
        .select(polygon=geo.polygons_from_rings_in_single_column("line"))
        .select(pl.col("polygon").repeat_by(2))
    )


def test_a_dtype_declares_its_crs() -> None:
    assert PointXY(crs=WGS84).ext_metadata() == '{"crs":"EPSG:4326"}'
    assert PointXY().ext_metadata() is None
    assert PointXY(crs=WGS84) != PointXY()
    assert isinstance(PointXY(crs=WGS84), PointXY)


def test_a_projjson_crs_is_written_as_an_object() -> None:
    """The spec asks for PROJJSON to be an object, not an escaped string."""
    metadata = json.loads(PointXY(crs=PROJJSON).ext_metadata())

    assert metadata == {"crs": json.loads(PROJJSON)}


def test_point_declares_its_crs(coords: pl.DataFrame, dimension: Dimension) -> None:
    df = coords.select(
        geo.point(
            "x",
            "y",
            z="z" if dimension.has_z else None,
            m="m" if dimension.has_m else None,
            crs=WGS84,
        ).alias("point")
    )

    assert df.schema["point"] == dimension.point_dtype(crs=WGS84)


@pytest.mark.parametrize(
    ("constructor", "nesting", "expected"),
    [
        (geo.line_string_from_columns, 1, LineStringXY),
        (geo.multi_point_from_columns, 1, MultiPointXY),
        (geo.polygons_from_coordinates, 1, PolygonXY),
        (geo.polygons_from_nested_coordinates, 2, PolygonXY),
        (geo.multi_line_string_from_columns, 2, MultiLineStringXY),
        (geo.multi_polygons_from_coordinates, 2, MultiPolygonXY),
        (geo.multi_polygons_from_nested_coordinates, 3, MultiPolygonXY),
    ],
    ids=[
        "linestring",
        "multipoint",
        "polygon",
        "nested polygon",
        "multilinestring",
        "multipolygon",
        "nested multipolygon",
    ],
)
def test_coordinate_columns_declare_a_crs(
    ring_coords: pl.DataFrame,
    constructor: Callable[..., pl.Expr],
    nesting: int,
    expected: type[GeoArrowType],
) -> None:
    grouped = ring_coords.group_by("polygon", "ring", maintain_order=True)
    nested = grouped.agg("x", "y")
    if nesting >= 2:
        nested = nested.group_by("polygon", maintain_order=True).agg("x", "y")
    if nesting == 3:
        nested = nested.select(pl.col("x", "y").implode())

    df = nested.select(geometry=constructor("x", "y", crs=WGS84))

    assert df.schema["geometry"] == expected(crs=WGS84)


@pytest.mark.parametrize(
    ("constructor", "parts", "expected"),
    [
        (geo.line_string_from_vertices, _points, LineStringXY),
        (geo.multi_point_from_points, _points, MultiPointXY),
        (geo.polygons_from_rings_in_single_column, _lines, PolygonXY),
        (geo.multi_line_string_from_line_strings, _lines, MultiLineStringXY),
        (geo.multi_polygons_from_polygons, _polygons, MultiPolygonXY),
    ],
    ids=["linestring", "multipoint", "polygon", "multilinestring", "multipolygon"],
)
class TestGatheringParts:
    def test_parts_without_a_crs_get_one(
        self,
        constructor: Callable[..., pl.Expr],
        parts: Callable[..., pl.DataFrame],
        expected: type[GeoArrowType],
    ) -> None:
        df = parts().select(geometry=constructor(pl.all(), crs=WGS84))

        assert df.schema["geometry"] == expected(crs=WGS84)

    def test_the_parts_crs_is_carried_over(
        self,
        constructor: Callable[..., pl.Expr],
        parts: Callable[..., pl.DataFrame],
        expected: type[GeoArrowType],
    ) -> None:
        df = parts(WGS84).select(geometry=constructor(pl.all()))

        assert df.schema["geometry"] == expected(crs=WGS84)

    def test_declaring_the_parts_crs_again_is_fine(
        self,
        constructor: Callable[..., pl.Expr],
        parts: Callable[..., pl.DataFrame],
        expected: type[GeoArrowType],
    ) -> None:
        df = parts(WGS84).select(geometry=constructor(pl.all(), crs=WGS84))

        assert df.schema["geometry"] == expected(crs=WGS84)

    def test_a_different_crs_is_refused_before_collecting(
        self,
        constructor: Callable[..., pl.Expr],
        parts: Callable[..., pl.DataFrame],
        expected: type[GeoArrowType],
    ) -> None:
        """Relabelling would silently move every coordinate somewhere else."""
        lf = parts(WGS84).lazy().select(constructor(pl.all(), crs="EPSG:28992"))

        with pytest.raises(ComputeError, match="is already"):
            lf.collect_schema()


def test_a_crs_is_set_next_to_the_parts_other_metadata() -> None:
    spherical = PointType.ext_from_params(
        "geoarrow.point", PointXY().ext_storage(), '{"edges":"spherical"}'
    )
    df = (
        pl.DataFrame({"x": [1.0], "y": [3.0]})
        .select(point=pl.struct("x", "y").ext.to(spherical))
        .select(pl.col("point").implode())
        .select(line=geo.line_string_from_vertices("point", crs=WGS84))
    )

    metadata = df.schema["line"].ext_metadata()
    assert metadata == '{"edges":"spherical","crs":"EPSG:4326"}'


def test_python_and_the_plugin_agree_byte_for_byte() -> None:
    """dtypes compare metadata as a string, so both sides have to write the same
    one, even for a CRS that is spelled with whitespace."""
    df = pl.DataFrame({"x": [[1.0]], "y": [[3.0]]}).select(
        line=geo.line_string_from_columns("x", "y", crs=PROJJSON)
    )

    assert df.schema["line"] == LineStringXY(crs=PROJJSON)


def test_the_dispatchers_and_namespace_pass_the_crs_on() -> None:
    df = _points().select(
        dispatched=geo.line_string("point", crs=WGS84),
        namespaced=gpl.col("point").geo.line_string(crs=WGS84),
        columns=geo.line_string(
            pl.lit(pl.Series([[1.0]])), pl.lit(pl.Series([[3.0]])), crs=WGS84
        ),
    )

    assert set(df.schema.values()) == {LineStringXY(crs=WGS84)}


def test_a_declared_crs_can_be_reprojected_from() -> None:
    df = pl.DataFrame({"x": [4.9041], "y": [52.3676]}).select(
        geo.to_crs(geo.point("x", "y", crs=WGS84), "EPSG:28992").alias("point")
    )

    assert df.schema["point"] == PointXY(crs="EPSG:28992")
    x, y = df["point"].ext.storage().struct.unnest().row(0)
    assert (round(x), round(y)) == (122097, 486745)


def test_a_target_crs_proj_does_not_know_is_refused_at_plan_time() -> None:
    lf = pl.LazyFrame({"x": [4.9041], "y": [52.3676]}).select(
        geo.to_crs(geo.point("x", "y", crs=WGS84), "EPSG:not-a-code")
    )

    with pytest.raises(ComputeError, match="cannot reproject from"):
        lf.collect_schema()


def test_a_column_without_a_crs_is_refused_for_reprojection_at_plan_time() -> None:
    lf = pl.LazyFrame({"x": [4.9041], "y": [52.3676]}).select(
        geo.to_crs(geo.point("x", "y"), WGS84)
    )

    with pytest.raises(ComputeError, match="does not declare a CRS"):
        lf.collect_schema()


def test_unknown_metadata_keywords_are_refused() -> None:
    with pytest.raises(TypeError):
        geo.point("x", "y", edges="spherical")  # type: ignore[call-arg]
