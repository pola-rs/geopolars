"""Unit tests for `geo.rotate`."""

from __future__ import annotations

import math
from typing import Literal

import polars as pl
import pytest
from polars.exceptions import ComputeError
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
    ring_coordinates,
)


def test_a_quarter_turn_is_exact(coords: pl.DataFrame, dimension: Dimension) -> None:
    """(x, y) becomes (-y, x) exactly, rather than off by a rounding error of cos(90°).
    Turning in the plane leaves `z` and `m` as they are."""
    df = coords.select(dimension.point())
    out = df.select(geo.rotate("point", 90))

    assert out.schema["point"] == dimension.point_dtype()
    expected = coordinates(df).with_columns(x=-pl.col("y"), y=pl.col("x"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize(
    ("amount", "unit"), [(90, "deg"), (math.pi / 2, "rad"), (450, "deg")]
)
def test_a_quarter_turn_leaves_no_rounding_error(amount: float, unit: str) -> None:
    """With cos(90°) computed, (1, 0) would land on (6e-17, 1)."""
    df = pl.DataFrame({"x": [1.0], "y": [0.0]}).select(XY.point())
    out = df.select(geo.rotate("point", amount, unit=unit))  # type: ignore[arg-type]

    expected = pl.DataFrame({"x": [0.0], "y": [1.0]})
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_a_negative_amount_turns_clockwise(coords: pl.DataFrame) -> None:
    df = coords.select(XY.point())
    out = df.select(geo.rotate("point", -90))

    expected = coordinates(df).select(x=pl.col("y"), y=-pl.col("x"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_a_full_turn_changes_nothing(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())

    assert_frame_equal(df.select(geo.rotate("point", 360)), df, check_exact=True)


def test_any_angle_turns_by_its_cosine_and_sine() -> None:
    df = pl.DataFrame({"x": [2.0], "y": [0.0]}).select(XY.point())
    out = df.select(geo.rotate("point", 30))

    expected = pl.DataFrame({"x": [2 * math.cos(math.pi / 6)], "y": [1.0]})
    assert_frame_equal(coordinates(out), expected)


@pytest.mark.parametrize("degrees", [90, 180, -45, 30])
def test_degrees_and_radians_agree(coords: pl.DataFrame, degrees: float) -> None:
    df = coords.select(XY.point())

    assert_frame_equal(
        df.select(geo.rotate("point", degrees)),
        df.select(geo.rotate("point", math.radians(degrees), unit="rad")),
    )


def test_turning_about_x_moves_y_into_z(coords: pl.DataFrame) -> None:
    df = coords.select(XYZM.point())
    out = df.select(geo.rotate("point", 90, axis="x"))

    expected = coordinates(df).with_columns(y=-pl.col("z"), z=pl.col("y"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_turning_about_y_moves_z_into_x(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point())
    out = df.select(geo.rotate("point", 90, axis="y"))

    expected = coordinates(df).with_columns(x=pl.col("z"), z=-pl.col("x"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


@pytest.mark.parametrize("dimension", [XY, XYM], ids=["PointXY", "PointXYM"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_turning_out_of_the_plane_needs_a_z_at_plan_time(
    coords: pl.DataFrame, dimension: Dimension, axis: str
) -> None:
    lf = coords.select(dimension.point()).lazy()

    with pytest.raises(TypeError, match=f"cannot rotate about the {axis} axis"):
        lf.select(geo.rotate("point", 90, axis=axis)).collect_schema()  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"unit": "pi"}, "`unit`"),
        ({"unit": "π"}, "`unit`"),
        ({"axis": "w"}, "`axis`"),
        ({"amount": math.nan}, "`amount`"),
        ({"amount": math.inf}, "`amount`"),
        ({"origin": (1.0,)}, "`origin`"),
        ({"origin": (1.0, 2.0, 3.0, 4.0)}, "`origin`"),
        ({"origin": (1.0, math.nan)}, "`origin`"),
    ],
)
def test_bad_arguments_are_refused(kwargs: dict[str, object], match: str) -> None:
    arguments: dict[str, object] = {"amount": 90.0, **kwargs}

    with pytest.raises(ValueError, match=match):
        geo.rotate("point", **arguments)  # type: ignore[arg-type]


def test_a_quarter_turn_about_an_origin(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """About (1, 1), (x, y) becomes (1 - (y - 1), 1 + (x - 1)): exact, like about (0, 0)."""
    df = coords.select(dimension.point())
    out = df.select(geo.rotate("point", 90, origin=(1.0, 1.0)))

    expected = coordinates(df).with_columns(x=2 - pl.col("y"), y=pl.col("x"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_the_origin_itself_stays_where_it_is() -> None:
    df = pl.DataFrame({"x": [5.0], "y": [7.0]}).select(XY.point())

    assert_frame_equal(df.select(geo.rotate("point", 33, origin=(5.0, 7.0))), df)


def test_turning_about_x_through_an_origin(coords: pl.DataFrame) -> None:
    """About the x axis through (0, 1, 1), (x, y, z) becomes (x, 1 - (z - 1), 1 + (y - 1))."""
    df = coords.select(XYZM.point())
    out = df.select(geo.rotate("point", 90, axis="x", origin=(0.0, 1.0, 1.0)))

    expected = coordinates(df).with_columns(y=2 - pl.col("z"), z=pl.col("y"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_a_two_dimensional_origin_is_at_z_zero(coords: pl.DataFrame) -> None:
    df = coords.select(XYZ.point())

    assert_frame_equal(
        df.select(geo.rotate("point", 30, axis="y", origin=(2.0, 3.0))),
        df.select(geo.rotate("point", 30, axis="y", origin=(2.0, 3.0, 0.0))),
    )


def test_every_vertex_of_every_ring_turns(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon turns as a whole, and keeps its rings as they were."""
    df = dimension.polygons(ring_coords)
    out = df.select(geo.rotate("polygon", 180))

    expected = ring_coordinates(df).with_columns(x=-pl.col("x"), y=-pl.col("y"))
    assert_frame_equal(ring_coordinates(out), expected, check_exact=True)
    assert_frame_equal(
        out.select(pl.col("polygon").ext.storage().list.eval(pl.element().list.len())),
        df.select(pl.col("polygon").ext.storage().list.eval(pl.element().list.len())),
    )


def test_keeps_empty_and_null_linestrings(dimension: Dimension) -> None:
    df = pl.DataFrame(
        {"vertices": [[], None]},
        schema={
            "vertices": pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
        },
    ).select(geo.line_string("vertices").alias("line"))

    assert_frame_equal(df.select(geo.rotate("line", 45)), df)


def test_the_namespace_forwards(line_coords: pl.DataFrame) -> None:
    df = XYZ.lines(line_coords)

    assert_frame_equal(
        line_coordinates(
            df.select(
                gpl.col("line").geo.rotate(math.pi / 4, "rad", "x", (1.0, 2.0, 3.0))
            )
        ),
        line_coordinates(
            df.select(geo.rotate("line", 45, axis="x", origin=(1.0, 2.0, 3.0)))
        ),
    )


def _origins(x: float, y: float, z: float | None = None) -> pl.Expr:
    """The same point for every row, as a column."""
    return geo.point(pl.lit(x), pl.lit(y), z=None if z is None else pl.lit(z))


@pytest.mark.parametrize(
    ("dimension", "axis"),
    [(XY, "z"), (XYM, "z"), (XYZ, "z"), (XYZM, "z"), (XYZ, "x"), (XYZM, "x")],
    ids=["XY", "XYM", "XYZ", "XYZM", "XYZ about x", "XYZM about x"],
)
def test_a_column_of_one_origin_is_the_same_as_its_numbers(
    ring_coords: pl.DataFrame, dimension: Dimension, axis: Literal["x", "z"]
) -> None:
    """The kernel that takes an origin per row, against the one that takes one origin."""
    df = dimension.polygons(ring_coords).with_columns(origin=_origins(1.5, -2.0, 3.0))

    assert_frame_equal(
        df.select(geo.rotate("polygon", 30, axis=axis, origin="origin")),
        df.select(geo.rotate("polygon", 30, axis=axis, origin=(1.5, -2.0, 3.0))),
        check_exact=True,
    )


def test_points_turn_about_a_column_the_same_as_about_its_numbers(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Points have no rows to look up: each coordinate is its own geometry."""
    df = coords.select(dimension.point(), origin=_origins(1.5, -2.0))

    assert_frame_equal(
        df.select(geo.rotate("point", 30, origin="origin")),
        df.select(geo.rotate("point", 30, origin=(1.5, -2.0))),
        check_exact=True,
    )


def test_each_geometry_turns_about_its_own_origin(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords).with_columns(
        origin=geo.mean_coordinate("polygon")
    )
    out = df.select(geo.rotate("polygon", 30, origin="origin"))

    for row, origin in enumerate(df["origin"].ext.storage().to_list()):
        alone = df.slice(row, 1).select(
            geo.rotate("polygon", 30, origin=(origin["x"], origin["y"]))
        )
        assert_frame_equal(out.slice(row, 1), alone, check_exact=True)


def test_turning_about_its_own_middle_keeps_the_middle(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords)
    turned = df.select(geo.rotate("polygon", 90, origin=geo.mean_coordinate("polygon")))

    assert_frame_equal(
        coordinates(turned.select(geo.mean_coordinate("polygon")), "polygon"),
        coordinates(df.select(geo.mean_coordinate("polygon")), "polygon"),
    )


def test_a_single_origin_is_used_for_every_row(line_coords: pl.DataFrame) -> None:
    df = XY.lines(line_coords)

    assert_frame_equal(
        df.select(geo.rotate("line", 30, origin=_origins(1.0, 2.0))),
        df.select(geo.rotate("line", 30, origin=(1.0, 2.0))),
        check_exact=True,
    )


def test_a_slice_turns_about_its_own_origins(ring_coords: pl.DataFrame) -> None:
    """A slice keeps the coordinates of the rows around it, as a streaming morsel does."""
    df = XYZ.polygons(ring_coords).with_columns(origin=geo.mean_coordinate("polygon"))
    turned = geo.rotate("polygon", 30, origin="origin")

    assert_frame_equal(
        df.slice(1, 1).select(turned), df.select(turned).slice(1, 1), check_exact=True
    )


def test_a_null_origin_gives_a_null_geometry(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords).with_columns(
        origin=pl.when(pl.int_range(pl.len()) == 0).then(geo.mean_coordinate("polygon"))
    )
    out = df.select(geo.rotate("polygon", 30, origin="origin"))

    assert out["polygon"].is_null().to_list() == [False, True]
    assert_frame_equal(
        out.slice(0, 1),
        df.slice(0, 1).select(geo.rotate("polygon", 30, origin="origin")),
    )


def test_a_null_origin_gives_a_null_point(coords: pl.DataFrame) -> None:
    df = coords.select(
        XY.point(), origin=geo.point(pl.col("x").replace(0.0, None), pl.col("y"))
    )
    out = df.select(geo.rotate("point", 30, origin="origin"))

    assert out["point"].is_null().to_list() == [False, False, True]


def test_keeps_empty_and_null_linestrings_about_a_column() -> None:
    df = pl.DataFrame(
        {"vertices": [[], None]},
        schema={"vertices": pl.List(pl.Struct(dict.fromkeys(XY.coords, pl.Float64)))},
    ).select(geo.line_string("vertices").alias("line"), origin=_origins(1.0, 2.0))

    assert_frame_equal(
        df.select(geo.rotate("line", 45, origin="origin")), df.select("line")
    )


def test_an_origin_column_has_to_hold_points(line_coords: pl.DataFrame) -> None:
    lf = XY.lines(line_coords).lazy()

    with pytest.raises(TypeError, match="`origin` has to be a `geoarrow.point`"):
        lf.select(geo.rotate("line", 30, origin="line")).collect_schema()
    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.select(geo.rotate("line", 30, origin=pl.lit(1.0))).collect_schema()


def test_an_origin_column_in_another_crs_is_refused(coords: pl.DataFrame) -> None:
    lf = coords.lazy().select(
        geo.point("x", "y", crs="EPSG:28992").alias("point"),
        origin=geo.point("x", "y", crs="EPSG:4326"),
    )

    with pytest.raises(ComputeError, match="use `to_crs`"):
        lf.select(geo.rotate("point", 30, origin="origin")).collect_schema()


def test_an_origin_column_without_a_crs_is_a_plain_position(
    coords: pl.DataFrame,
) -> None:
    df = coords.select(
        geo.point("x", "y", crs="EPSG:28992").alias("point"), origin=_origins(1.0, 2.0)
    )

    assert_frame_equal(
        df.select(geo.rotate("point", 30, origin="origin")),
        df.select(geo.rotate("point", 30, origin=(1.0, 2.0))),
    )


@pytest.mark.parametrize("dimension", [XY, XYM], ids=["PointXY", "PointXYM"])
def test_turning_out_of_the_plane_about_a_column_needs_a_z(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    lf = coords.lazy().select(dimension.point(), origin=_origins(1.0, 2.0, 3.0))

    with pytest.raises(TypeError, match="cannot rotate about the x axis"):
        lf.select(geo.rotate("point", 30, axis="x", origin="origin")).collect_schema()


def test_the_namespace_forwards_an_origin_column(line_coords: pl.DataFrame) -> None:
    df = XYZ.lines(line_coords).with_columns(origin=geo.mean_coordinate("line"))

    assert_frame_equal(
        df.select(gpl.col("line").geo.rotate(30, origin="origin")),
        df.select(geo.rotate("line", 30, origin="origin")),
    )
