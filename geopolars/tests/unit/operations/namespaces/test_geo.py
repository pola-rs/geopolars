"""Unit tests for the `geo` namespace"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from tests.unit.conftest import (
    XY,
    XYM,
    XYZ,
    XYZM,
    Dimension,
    coordinates,
    line_coordinates,
    multilinestring_coordinates,
    multipoint_coordinates,
    multipolygon_coordinates,
    ring_coordinates,
)


def test_translate_shifts_x_and_y(coords: pl.DataFrame, dimension: Dimension) -> None:
    df = coords.select(dimension.point())
    out = df.select(geo.translate("point", (1.5, -2.0)).alias("point"))

    assert_frame_equal(
        coordinates(out),
        coordinates(df).with_columns(pl.col("x") + 1.5, pl.col("y") - 2.0),
    )


def test_translate_keeps_the_dimension(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Moving a point does not change what kind of point it is."""
    df = coords.select(dimension.point())
    out = df.select(geo.translate("point", (1.0, 1.0)).alias("point"))

    assert out.schema["point"] == dimension.point_dtype()


def test_translate_by_zero_is_the_identity(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())
    out = df.select(geo.translate("point", (0.0, 0.0)).alias("point"))

    assert_frame_equal(out, df)


def test_translate_shifts_z(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point())
    out = df.select(geo.translate("point", (0.0, 0.0, 5.0)).alias("point"))

    assert_frame_equal(
        coordinates(out), coordinates(df).with_columns(pl.col("z") + 5.0)
    )


def test_translate_leaves_m_untouched(coords: pl.DataFrame) -> None:
    """`m` is a measure, not a position: moving the geometry must not change
    the timestamp or distance-along-route the vertex carries."""
    df = coords.select(XYZM.point())
    out = df.select(geo.translate("point", (10.0, 10.0, 10.0)).alias("point"))

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

    with pytest.raises(TypeError, match="cannot translate by a z offset"):
        df.select(geo.translate("point", (0.0, 0.0, 1.0)))


def test_translate_rejects_dz_at_plan_time(coords: pl.DataFrame) -> None:
    lf = coords.select(XY.point()).lazy()

    with pytest.raises(TypeError, match="cannot translate by a z offset"):
        lf.select(geo.translate("point", (0.0, 0.0, 1.0))).collect_schema()


def test_translate_rejects_a_plain_float_column() -> None:
    df = pl.DataFrame({"point": [1.0, 2.0]})

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        df.select(geo.translate("point", (1.0, 1.0)))


def test_translate_rejects_a_bare_coordinate_struct() -> None:
    df = pl.DataFrame({"point": [{"x": 1.0, "y": 2.0}]})

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        df.select(geo.translate("point", (1.0, 1.0)))


def test_translate_keeps_a_null_point_null() -> None:
    """Shifting the coordinates rebuilds the storage struct, and the row that
    says 'no point here' must survive that: GeoArrow has no point with null
    coordinates, only a null point."""
    df = pl.DataFrame({"x": [None, 1.0], "y": [2.0, 2.0]}).select(
        geo.point("x", "y").alias("point")
    )
    out = df.select(geo.translate("point", (1.0, 1.0)).alias("point"))

    assert out["point"].is_null().to_list() == [True, False]


def test_translate_shifts_every_vertex(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A linestring moves as a whole: the same offset applies to each of its
    vertices, and which vertices belong to which line does not change."""
    df = dimension.lines(line_coords)
    out = df.select(geo.translate("line", (1.5, -2.0)).alias("line"))

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
    out = df.select(geo.translate("line", (10.0, 10.0, 10.0)).alias("line"))

    assert_frame_equal(
        line_coordinates(out).select("m"), line_coordinates(df).select("m")
    )


def test_translate_rejects_dz_on_a_linestring_with_no_z(
    line_coords: pl.DataFrame,
) -> None:
    df = XY.lines(line_coords)

    with pytest.raises(TypeError, match="cannot translate by a z offset"):
        df.select(geo.translate("line", (0.0, 0.0, 1.0)))


def test_translate_keeps_empty_and_null_linestrings(dimension: Dimension) -> None:
    """An empty line has nothing to shift and a null one is still null."""
    df = pl.DataFrame(
        {"vertices": [[], None]},
        schema={
            "vertices": pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
        },
    ).select(geo.line_string("vertices").alias("line"))
    out = df.select(geo.translate("line", (1.0, 1.0)).alias("line"))

    assert_frame_equal(out, df)


def test_translate_shifts_every_point_of_a_multipoint(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)
    out = df.select(geo.translate("multipoint", (1.5, -2.0)).alias("multipoint"))

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
    out = df.select(geo.translate("multipoint", (10.0, 10.0, 10.0)).alias("multipoint"))

    assert_frame_equal(
        multipoint_coordinates(out).select("m"),
        multipoint_coordinates(df).select("m"),
    )


def test_translate_rejects_dz_on_a_multipoint_with_no_z(
    line_coords: pl.DataFrame,
) -> None:
    df = XY.multipoints(line_coords)

    with pytest.raises(TypeError, match="cannot translate by a z offset"):
        df.select(geo.translate("multipoint", (0.0, 0.0, 1.0)))


def test_translate_shifts_every_vertex_of_every_linestring(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multilinestrings(ring_coords)
    out = df.select(
        geo.translate("multilinestring", (1.5, -2.0)).alias("multilinestring")
    )
    parts = pl.col("multilinestring").ext.storage()

    assert out.schema["multilinestring"] == dimension.multilinestring_dtype()
    assert_frame_equal(
        multilinestring_coordinates(out),
        multilinestring_coordinates(df).with_columns(
            pl.col("x") + 1.5, pl.col("y") - 2.0
        ),
    )
    assert_frame_equal(out.select(parts.list.len()), df.select(parts.list.len()))
    assert_frame_equal(
        out.select(parts.explode(empty_as_null=False).list.len()),
        df.select(parts.explode(empty_as_null=False).list.len()),
    )


def test_translate_rejects_dz_on_a_multilinestring_with_no_z(
    ring_coords: pl.DataFrame,
) -> None:
    df = XY.multilinestrings(ring_coords)

    with pytest.raises(TypeError, match="cannot translate by a z offset"):
        df.select(geo.translate("multilinestring", (0.0, 0.0, 1.0)))


def test_translate_keeps_empty_and_null_multilinestrings(
    dimension: Dimension,
) -> None:
    df = pl.DataFrame(
        {"lines": [[], [[]], None]},
        schema={
            "lines": pl.List(
                pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
            )
        },
    ).select(geo.multi_line_string("lines").alias("multilinestring"))
    out = df.select(
        geo.translate("multilinestring", (1.0, 1.0)).alias("multilinestring")
    )

    assert_frame_equal(out, df)


def test_translate_shifts_every_vertex_of_every_ring(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon moves as a whole."""
    df = dimension.polygons(ring_coords)
    out = df.select(geo.translate("polygon", (1.5, -2.0)).alias("polygon"))
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


def test_translate_keeps_empty_and_null_polygons(dimension: Dimension) -> None:
    df = pl.DataFrame(
        {"rings": [[], [[]], None]},
        schema={
            "rings": pl.List(
                pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
            )
        },
    ).select(geo.polygons_from_rings_in_single_column("rings").alias("polygon"))
    out = df.select(geo.translate("polygon", (1.0, 1.0)).alias("polygon"))

    assert_frame_equal(out, df)


def test_translate_shifts_every_vertex_of_every_polygon(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A multipolygon moves as a whole, and keeps its polygons and rings."""
    df = dimension.multipolygons(multipolygon_coords)
    out = df.select(geo.translate("multipolygon", (1.5, -2.0)).alias("multipolygon"))
    polygons = pl.col("multipolygon").ext.storage()
    rings = polygons.explode(empty_as_null=False)

    assert out.schema["multipolygon"] == dimension.multipolygon_dtype()
    assert_frame_equal(
        multipolygon_coordinates(out),
        multipolygon_coordinates(df).with_columns(pl.col("x") + 1.5, pl.col("y") - 2.0),
    )
    for structure in (polygons, rings, rings.explode(empty_as_null=False)):
        assert_frame_equal(
            out.select(structure.list.len()), df.select(structure.list.len())
        )


def test_translate_rejects_dz_on_a_multipolygon_with_no_z(
    multipolygon_coords: pl.DataFrame,
) -> None:
    df = XY.multipolygons(multipolygon_coords)

    with pytest.raises(TypeError, match="cannot translate by a z offset"):
        df.select(geo.translate("multipolygon", (0.0, 0.0, 1.0)))


def test_translate_keeps_empty_and_null_multipolygons(
    dimension: Dimension,
) -> None:
    df = pl.DataFrame(
        {"polygons": [[], [[]], [[[]]], None]},
        schema={
            "polygons": pl.List(
                pl.List(pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64))))
            )
        },
    ).select(geo.multi_polygons_from_polygons("polygons").alias("multipolygon"))
    out = df.select(geo.translate("multipolygon", (1.0, 1.0)).alias("multipolygon"))

    assert_frame_equal(out, df)


def test_namespace_matches_the_functional_api(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point())

    assert_frame_equal(
        df.select(gpl.col("point").geo.translate((1.0, 2.0, 3.0)).alias("point")),
        df.select(geo.translate("point", (1.0, 2.0, 3.0)).alias("point")),
    )


def test_linestring_namespace_matches_the_functional_api(
    line_coords: pl.DataFrame,
) -> None:
    vertices = line_coords.group_by("line", maintain_order=True).agg(XYZ.point())

    assert_frame_equal(
        vertices.select(gpl.col("point").geo.line_string().alias("line")),
        vertices.select(geo.line_string("point").alias("line")),
    )


def test_multipoint_namespace_matches_the_functional_api(
    line_coords: pl.DataFrame,
) -> None:
    points = line_coords.group_by("line", maintain_order=True).agg(XYZ.point())

    assert_frame_equal(
        points.select(gpl.col("point").geo.multi_point().alias("multipoint")),
        points.select(geo.multi_point("point").alias("multipoint")),
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
        rings.select(
            gpl.col("line").geo.polygons_from_rings_in_single_column().alias("polygon")
        ),
        rings.select(geo.polygons_from_rings_in_single_column("line").alias("polygon")),
    )


def test_separate_rings_namespace_matches_the_functional_api(
    ring_coords: pl.DataFrame,
) -> None:
    lines = (
        ring_coords.group_by("polygon", "ring", maintain_order=True)
        .agg(XYZ.point())
        .select("polygon", "ring", XYZ.linestring())
    )
    exteriors = lines.filter(pl.col("ring") == 0).select("polygon", exterior="line")
    holes = lines.filter(pl.col("ring") > 0).group_by("polygon").agg(holes="line")
    rings = exteriors.join(holes, on="polygon", how="left")

    assert_frame_equal(
        rings.select(
            gpl.col("exterior")
            .geo.polygons_from_rings_in_separate_columns("holes")
            .alias("polygon")
        ),
        rings.select(
            geo.polygons_from_rings_in_separate_columns("exterior", "holes").alias(
                "polygon"
            )
        ),
    )


def test_polygon_coordinates_namespace_matches_the_functional_api(
    ring_coords: pl.DataFrame,
) -> None:
    exteriors = (
        ring_coords.filter(pl.col("ring") == 0)
        .group_by("polygon", maintain_order=True)
        .agg("x", "y", "z")
    )

    assert_frame_equal(
        exteriors.select(gpl.col("x").geo.polygons_from_coordinates("y", "z")),
        exteriors.select(geo.polygons_from_coordinates("x", "y", "z")),
    )


def test_polygon_nested_coordinates_namespace_matches_the_functional_api(
    ring_coords: pl.DataFrame,
) -> None:
    rings = (
        ring_coords.group_by("polygon", "ring", maintain_order=True)
        .agg("x", "y", "z")
        .group_by("polygon", maintain_order=True)
        .agg("x", "y", "z")
    )

    assert_frame_equal(
        rings.select(gpl.col("x").geo.polygons_from_nested_coordinates("y", "z")),
        rings.select(geo.polygons_from_nested_coordinates("x", "y", "z")),
    )


def test_multilinestring_namespace_matches_the_functional_api(
    ring_coords: pl.DataFrame,
) -> None:
    lines = (
        ring_coords.group_by("polygon", "ring", maintain_order=True)
        .agg(XYZ.point())
        .select("polygon", XYZ.linestring())
        .group_by("polygon", maintain_order=True)
        .agg("line")
    )

    assert_frame_equal(
        lines.select(gpl.col("line").geo.multi_line_string().alias("multilinestring")),
        lines.select(geo.multi_line_string("line").alias("multilinestring")),
    )


def test_multipolygon_namespace_matches_the_functional_api(
    multipolygon_coords: pl.DataFrame,
) -> None:
    polygons = (
        multipolygon_coords.group_by(
            "multipolygon", "polygon", "ring", maintain_order=True
        )
        .agg(XYZ.point())
        .select("multipolygon", "polygon", XYZ.linestring())
        .group_by("multipolygon", "polygon", maintain_order=True)
        .agg("line")
        .select("multipolygon", XYZ.polygon())
        .group_by("multipolygon", maintain_order=True)
        .agg("polygon")
    )

    assert_frame_equal(
        polygons.select(
            gpl.col("polygon").geo.multi_polygons_from_polygons().alias("multipolygon")
        ),
        polygons.select(
            geo.multi_polygons_from_polygons("polygon").alias("multipolygon")
        ),
    )


def test_multipolygon_coordinates_namespace_matches_the_functional_api(
    multipolygon_coords: pl.DataFrame,
) -> None:
    exteriors = (
        multipolygon_coords.filter(pl.col("ring") == 0)
        .group_by("multipolygon", "polygon", maintain_order=True)
        .agg("x", "y", "z")
        .group_by("multipolygon", maintain_order=True)
        .agg("x", "y", "z")
    )

    assert_frame_equal(
        exteriors.select(gpl.col("x").geo.multi_polygons_from_coordinates("y", "z")),
        exteriors.select(geo.multi_polygons_from_coordinates("x", "y", "z")),
    )


def test_multipolygon_nested_coordinates_namespace_matches_the_functional_api(
    multipolygon_coords: pl.DataFrame,
) -> None:
    rings = (
        multipolygon_coords.group_by(
            "multipolygon", "polygon", "ring", maintain_order=True
        )
        .agg("x", "y", "z")
        .group_by("multipolygon", "polygon", maintain_order=True)
        .agg("x", "y", "z")
        .group_by("multipolygon", maintain_order=True)
        .agg("x", "y", "z")
    )

    assert_frame_equal(
        rings.select(gpl.col("x").geo.multi_polygons_from_nested_coordinates("y", "z")),
        rings.select(geo.multi_polygons_from_nested_coordinates("x", "y", "z")),
    )


def test_multipolygon_namespace_forwards_close() -> None:
    """An open ring is only kept when `close` reaches the constructor."""
    open_ring = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 1.0}]
    vertices = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
    df = pl.DataFrame(
        {"polygons": [[[open_ring]]]}, schema={"polygons": pl.List(pl.List(vertices))}
    )
    close = geo.CloseRings()

    namespaced = df.select(
        gpl.col("polygons").geo.multi_polygons_from_polygons(close=close)
    )
    functional = df.select(geo.multi_polygons_from_polygons("polygons", close=close))

    assert namespaced["polygons"].is_null().to_list() == [False]
    assert_frame_equal(namespaced, functional)


def test_set_crs_namespace_matches_the_functional_api() -> None:
    points = pl.DataFrame({"x": [4.9], "y": [52.4]}).select(
        geo.point("x", "y", crs="EPSG:4326").alias("point")
    )

    assert_frame_equal(
        points.select(gpl.col("point").geo.set_crs("EPSG:28992", force=True)),
        points.select(geo.set_crs("point", "EPSG:28992", force=True)),
    )


def test_crs_to_projjson_namespace_matches_the_functional_api() -> None:
    points = pl.DataFrame({"x": [4.9], "y": [52.4]}).select(
        geo.point("x", "y", crs="EPSG:4326").alias("point")
    )

    assert_frame_equal(
        points.select(gpl.col("point").geo.crs_to_projjson()),
        points.select(geo.crs_to_projjson("point")),
    )


def test_is_geographic_namespace_matches_the_functional_api() -> None:
    points = pl.DataFrame({"x": [4.9], "y": [52.4]}).select(
        geo.point("x", "y", crs="EPSG:4326").alias("point")
    )

    assert_frame_equal(
        points.select(gpl.col("point").geo.is_geographic(ignore_errors=True)),
        points.select(geo.is_geographic("point", ignore_errors=True)),
    )


def test_to_crs_namespace_matches_the_functional_api(
    line_coords: pl.DataFrame,
) -> None:
    lines = (
        line_coords.group_by("line", maintain_order=True)
        .agg("x", "y", "z")
        .select(line=geo.line_string_from_columns("x", "y", z="z", crs="EPSG:4326"))
    )

    # Neither CRS specifies z, so it is kept as it is.
    flag = {"allow_projection_with_unscaled_z": True}
    assert_frame_equal(
        lines.select(gpl.col("line").geo.to_crs("EPSG:3857", **flag)),
        lines.select(geo.to_crs("line", "EPSG:3857", **flag)),
    )


def test_translate_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    """The output type is derived from the input's dtype,
    so a bad column is a schema error."""
    lf = pl.LazyFrame({"lon": [1.0]}).select(geo.translate("lon", (1.0, 1.0)))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()
