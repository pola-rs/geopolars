"""The bounding box of a geometry."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import BoxType, BoxXY, PolygonXY
from tests.unit.conftest import XY, Dimension

_INF = float("inf")
_NAN = float("nan")


def _box_dtype(dimension: Dimension) -> type[BoxType]:
    return BoxType.of_dimension(dimension.coords)


def _bounds_per(
    vertices: pl.DataFrame, group: str, dimension: Dimension
) -> pl.DataFrame:
    """The bounds per geometry, straight off a vertex frame, in box field order."""
    coords = dimension.coords
    return (
        vertices.group_by(group, maintain_order=True)
        .agg(
            *(pl.col(c).min().alias(f"{c}min") for c in coords),
            *(pl.col(c).max().alias(f"{c}max") for c in coords),
        )
        .drop(group)
    )


def _unnest(df: pl.DataFrame, name: str) -> pl.DataFrame:
    return df.select(pl.col(name).ext.storage()).to_series().struct.unnest()


def test_a_point_repeats_its_own_coordinates(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = coords.select(dimension.point()).select(geo.bounds("point"))

    assert out.schema["point"] == _box_dtype(dimension)()
    expected = coords.select(
        *(pl.col(c).alias(f"{c}min") for c in dimension.coords),
        *(pl.col(c).alias(f"{c}max") for c in dimension.coords),
    )
    assert_frame_equal(_unnest(out, "point"), expected)


def test_a_linestring_is_bounded_by_its_vertices(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.lines(line_coords).select(geo.bounds("line"))

    assert out.schema["line"] == _box_dtype(dimension)()
    assert_frame_equal(
        _unnest(out, "line"), _bounds_per(line_coords, "line", dimension)
    )


def test_a_polygon_is_bounded_by_its_rings(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.polygons(ring_coords).select(geo.bounds("polygon"))

    assert_frame_equal(
        _unnest(out, "polygon"), _bounds_per(ring_coords, "polygon", dimension)
    )


def test_a_multipoint_gets_one_box_around_all_its_points(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.multipoints(line_coords).select(geo.bounds("multipoint"))

    assert out.schema["multipoint"] == _box_dtype(dimension)()
    assert_frame_equal(
        _unnest(out, "multipoint"), _bounds_per(line_coords, "line", dimension)
    )


def test_a_multilinestring_gets_one_box_around_all_its_linestrings(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.multilinestrings(ring_coords).select(geo.bounds("multilinestring"))

    assert out.schema["multilinestring"] == _box_dtype(dimension)()
    assert_frame_equal(
        _unnest(out, "multilinestring"), _bounds_per(ring_coords, "polygon", dimension)
    )


def test_a_multipolygon_gets_one_box_around_all_its_polygons(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    out = dimension.multipolygons(multipolygon_coords).select(
        geo.bounds("multipolygon")
    )

    assert out.schema["multipolygon"] == _box_dtype(dimension)()
    assert_frame_equal(
        _unnest(out, "multipolygon"),
        _bounds_per(multipolygon_coords, "multipolygon", dimension),
    )


def test_margins_widen_x_and_y_on_both_sides_only(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    lines = dimension.lines(line_coords)
    plain = _unnest(lines.select(geo.bounds("line")), "line")
    padded = _unnest(
        lines.select(geo.bounds("line", margin_x=1.0, margin_y=2.0)), "line"
    )

    expected = plain.with_columns(
        pl.col("xmin") - 1.0,
        pl.col("xmax") + 1.0,
        pl.col("ymin") - 2.0,
        pl.col("ymax") + 2.0,
    )
    assert_frame_equal(padded, expected)


def test_a_margin_around_a_point() -> None:
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(geo.point("x", "y"))
    out = df.select(geo.bounds("x", margin_x=0.5, margin_y=1.0))

    assert out["x"].ext.storage().to_list() == [
        {"xmin": 0.5, "ymin": 1.0, "xmax": 1.5, "ymax": 3.0}
    ]


def test_a_margin_widens_the_box_of_a_multigeometry(line_coords: pl.DataFrame) -> None:
    df = XY.multipoints(line_coords)
    plain = _unnest(df.select(geo.bounds("multipoint")), "multipoint")
    padded = _unnest(df.select(geo.bounds("multipoint", margin_x=1.0)), "multipoint")

    assert_frame_equal(
        padded, plain.with_columns(pl.col("xmin") - 1.0, pl.col("xmax") + 1.0)
    )


_EMPTY = {"xmin": _INF, "ymin": _INF, "xmax": -_INF, "ymax": -_INF}


def test_an_empty_point_has_an_empty_box() -> None:
    df = pl.DataFrame({"x": [_NAN], "y": [_NAN]}).select(geo.point("x", "y"))
    out = df.select(geo.bounds("x", margin_x=1.0))

    assert out["x"].ext.storage().to_list() == [_EMPTY]


def test_empty_and_null_geometries() -> None:
    storage = pl.List(pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64})))
    df = pl.DataFrame(
        {"polygon": [[], [[]], None]}, schema={"polygon": storage}
    ).select(pl.col("polygon").ext.to(PolygonXY()))
    out = df.select(geo.bounds("polygon"))

    assert out["polygon"].ext.storage().to_list() == [_EMPTY, _EMPTY, None]


def test_empty_points_are_skipped() -> None:
    # One linestring through an empty point and two real ones.
    df = pl.DataFrame({"x": [[_NAN, 1.0, 3.0]], "y": [[_NAN, 2.0, 4.0]]}).select(
        geo.line_string("x", "y")
    )
    out = df.select(geo.bounds("x"))

    assert out["x"].ext.storage().to_list() == [
        {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0}
    ]


def test_a_null_multigeometry_stays_null() -> None:
    df = pl.DataFrame({"x": [[1.0], None], "y": [[2.0], None]}).select(
        geo.multi_point("x", "y")
    )
    out = df.select(geo.bounds("x"))

    assert out["x"].ext.storage().to_list() == [
        {"xmin": 1.0, "ymin": 2.0, "xmax": 1.0, "ymax": 2.0},
        None,
    ]


def test_the_crs_is_carried_over() -> None:
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(
        geo.point("x", "y", crs="EPSG:4326")
    )

    assert df.select(geo.bounds("x")).schema["x"] == BoxXY(crs="EPSG:4326")
    multipoints = df.select(geo.multi_point(pl.col("x").implode()))
    assert multipoints.select(geo.bounds("x")).schema["x"] == BoxXY(crs="EPSG:4326")


def test_a_rectangle_round_trips_through_its_box() -> None:
    ring = {"x": [0.0, 2.0, 2.0, 0.0, 0.0], "y": [1.0, 1.0, 3.0, 3.0, 1.0]}
    df = pl.DataFrame({k: [v] for k, v in ring.items()}).select(
        geo.polygons_from_coordinates("x", "y")
    )
    out = df.select(geo.to_polygon(geo.bounds("x")))

    assert_frame_equal(out, df)


@pytest.mark.parametrize(
    "column",
    [
        pl.Series("x", [1.0]),
        pl.Series("x", [{"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}]),
    ],
)
def test_rejects_a_non_geometry_at_plan_time(column: pl.Series) -> None:
    lf = column.to_frame().lazy()
    if column.dtype != pl.Float64:
        lf = lf.select(pl.col("x").ext.to(BoxXY()))
    with pytest.raises(TypeError):
        lf.select(geo.bounds("x")).collect_schema()


def test_namespace_matches_the_functional_api() -> None:
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(geo.point("x", "y").alias("p"))

    assert_frame_equal(
        df.select(gpl.col("p").geo.bounds(margin_x=1.0, margin_y=2.0)),
        df.select(geo.bounds("p", margin_x=1.0, margin_y=2.0)),
    )


def _x_bounds(xs: list[list[float]], crs: str | None, margin_x: float = 0.0) -> list:
    """`(xmin, xmax)` of the bounds of one linestring per list of `x`."""
    df = pl.DataFrame(
        {"x": xs, "y": [[0.0] * len(x) for x in xs]},
        schema={"x": pl.List(pl.Float64), "y": pl.List(pl.Float64)},
    ).select(geo.line_string("x", "y", crs=crs))
    out = df.select(geo.bounds("x", margin_x=margin_x)).to_series()
    return out.ext.storage().struct.unnest().select("xmin", "xmax").rows()


@pytest.mark.parametrize(
    ("xs", "expected"),
    [
        ([170.0, 190.0], (170.0, -170.0)),
        ([-190.0, -170.0], (170.0, -170.0)),
        ([175.0, 179.0], (175.0, 179.0)),
        ([-170.0, 180.0], (-170.0, 180.0)),
        ([-180.0, 180.0], (-180.0, 180.0)),
        ([0.0, 400.0], (-180.0, 180.0)),
        ([540.0, 545.0], (-180.0, -175.0)),
        ([180.0], (-180.0, -180.0)),
    ],
)
def test_a_geographic_box_crosses_the_antimeridian(
    xs: list[float], expected: tuple[float, float]
) -> None:
    assert _x_bounds([xs], "EPSG:4326") == [expected]


def test_a_margin_can_push_a_box_over_the_antimeridian() -> None:
    assert _x_bounds([[175.0, 179.0]], "EPSG:4326", margin_x=2.0) == [(173.0, -179.0)]


def test_a_margin_can_cover_the_globe() -> None:
    assert _x_bounds([[-179.0, 179.0]], "EPSG:4326", margin_x=1.0) == [(-180.0, 180.0)]


def test_without_a_crs_bounds_are_left_past_the_edge() -> None:
    assert _x_bounds([[170.0, 190.0], [0.0, 400.0]], None) == [
        (170.0, 190.0),
        (0.0, 400.0),
    ]


def test_a_multigeometry_crosses_the_antimeridian_as_a_whole() -> None:
    df = pl.DataFrame({"x": [[10.0, 190.0]], "y": [[0.0, 0.0]]}).select(
        geo.multi_point("x", "y", crs="EPSG:4326")
    )
    boxes = df.select(geo.bounds("x", margin_x=15.0)).to_series()

    assert [(box["xmin"], box["xmax"]) for box in boxes.ext.storage().to_list()] == [
        (-5.0, -155.0)
    ]


def test_an_empty_geographic_geometry_has_an_empty_box() -> None:
    assert _x_bounds([[]], "EPSG:4326", margin_x=1.0) == [(_INF, -_INF)]


def test_a_crossing_box_turns_back_into_the_polygon_past_the_antimeridian() -> None:
    ring = {"x": [170.0, 190.0, 190.0, 170.0, 170.0], "y": [0.0, 0.0, 1.0, 1.0, 0.0]}
    df = pl.DataFrame({k: [v] for k, v in ring.items()}).select(
        geo.polygons_from_coordinates("x", "y", crs="EPSG:4326")
    )

    assert_frame_equal(df.select(geo.to_polygon(geo.bounds("x"))), df)
