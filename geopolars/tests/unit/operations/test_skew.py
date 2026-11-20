"""Unit tests for `geo.skew`."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from tests.unit.conftest import XY, Dimension, coordinates, ring_coordinates


def test_xs_leans_along_x(coords: pl.DataFrame, dimension: Dimension) -> None:
    """At 45°, every position moves along `x` by exactly its `y`.
    `z` and `m` stay as they are."""
    df = coords.select(dimension.point())
    out = df.select(geo.skew("point", xs=45))

    assert out.schema["point"] == dimension.point_dtype()
    expected = coordinates(df).with_columns(x=pl.col("x") + pl.col("y"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_ys_leans_along_y(coords: pl.DataFrame, dimension: Dimension) -> None:
    df = coords.select(dimension.point())
    out = df.select(geo.skew("point", ys=-45))

    expected = coordinates(df).with_columns(y=pl.col("y") - pl.col("x"))
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_both_angles_apply_to_the_positions_as_they_were(coords: pl.DataFrame) -> None:
    df = coords.select(XY.point())
    out = df.select(geo.skew("point", xs=45, ys=45))

    expected = coordinates(df).select(
        x=pl.col("x") + pl.col("y"), y=pl.col("y") + pl.col("x")
    )
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_no_angle_changes_nothing(coords: pl.DataFrame, dimension: Dimension) -> None:
    df = coords.select(dimension.point())

    assert_frame_equal(df.select(geo.skew("point")), df, check_exact=True)


def test_any_angle_leans_by_its_tangent() -> None:
    df = pl.DataFrame({"x": [1.0], "y": [2.0]}).select(XY.point())
    out = df.select(geo.skew("point", xs=30, ys=10))

    expected = pl.DataFrame(
        {
            "x": [1.0 + math.tan(math.radians(30)) * 2.0],
            "y": [2.0 + math.tan(math.radians(10)) * 1.0],
        }
    )
    assert_frame_equal(coordinates(out), expected)


@pytest.mark.parametrize("degrees", [45, -45, 135, 30])
def test_degrees_and_radians_agree(coords: pl.DataFrame, degrees: float) -> None:
    df = coords.select(XY.point())
    radians = math.radians(degrees)

    assert_frame_equal(
        df.select(geo.skew("point", xs=degrees, ys=degrees)),
        df.select(geo.skew("point", xs=radians, ys=radians, unit="rad")),
    )


def test_an_eighth_turn_leaves_no_rounding_error() -> None:
    """With tan(45°) computed, the factor would be 0.9999999999999999."""
    df = pl.DataFrame({"x": [0.0], "y": [3.0]}).select(XY.point())
    out = df.select(geo.skew("point", xs=45))

    expected = pl.DataFrame({"x": [3.0], "y": [3.0]})
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_skewing_about_an_origin(coords: pl.DataFrame, dimension: Dimension) -> None:
    """About (2, 1), `x` moves by how far `y` is from 1."""
    df = coords.select(dimension.point())
    out = df.select(geo.skew("point", xs=45, origin=(2.0, 1.0)))

    expected = coordinates(df).with_columns(x=pl.col("x") + pl.col("y") - 1)
    assert_frame_equal(coordinates(out), expected, check_exact=True)


def test_the_origin_itself_stays_where_it_is() -> None:
    df = pl.DataFrame({"x": [5.0], "y": [7.0]}).select(XY.point())

    assert_frame_equal(
        df.select(geo.skew("point", xs=20, ys=-35, origin=(5.0, 7.0))), df
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"xs": 90}, "`xs` cannot be a quarter turn"),
        ({"ys": -90}, "`ys` cannot be a quarter turn"),
        ({"xs": 270}, "`xs` cannot be a quarter turn"),
        ({"xs": math.pi / 2, "unit": "rad"}, "`xs` cannot be a quarter turn"),
        ({"xs": math.nan}, "`xs`"),
        ({"ys": math.inf}, "`ys`"),
        ({"unit": "pi"}, "`unit`"),
        ({"origin": (1.0,)}, "`origin`"),
    ],
)
def test_bad_arguments_are_refused(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        geo.skew("point", **kwargs)  # type: ignore[arg-type]


def test_refuses_a_column_that_is_not_a_geometry() -> None:
    df = pl.DataFrame({"point": [1.0, 2.0]})

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        df.lazy().select(geo.skew("point", xs=10)).collect_schema()


def test_every_vertex_of_every_ring_leans(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon is sheared as a whole, and keeps its rings as they were."""
    df = dimension.polygons(ring_coords)
    out = df.select(geo.skew("polygon", ys=45))

    expected = ring_coordinates(df).with_columns(y=pl.col("y") + pl.col("x"))
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

    assert_frame_equal(df.select(geo.skew("line", xs=30)), df)


def test_the_namespace_forwards(coords: pl.DataFrame) -> None:
    df = coords.select(XY.point())

    assert_frame_equal(
        df.select(
            gpl.col("point").geo.skew(
                math.radians(18), math.radians(-36), "rad", (1.0, 2.0)
            )
        ),
        df.select(geo.skew("point", xs=18, ys=-36, origin=(1.0, 2.0))),
    )


def _origins(x: float, y: float) -> pl.Expr:
    """The same point for every row, as a column."""
    return geo.point(pl.lit(x), pl.lit(y))


@pytest.mark.parametrize(("xs", "ys"), [(45, 0), (30, -10)], ids=["exact", "rounded"])
def test_a_column_of_one_origin_is_the_same_as_its_numbers(
    ring_coords: pl.DataFrame, dimension: Dimension, xs: float, ys: float
) -> None:
    """The kernel that takes an origin per row, against the one that takes one origin."""
    df = dimension.polygons(ring_coords).with_columns(origin=_origins(1.5, -2.0))

    assert_frame_equal(
        df.select(geo.skew("polygon", xs, ys, origin="origin")),
        df.select(geo.skew("polygon", xs, ys, origin=(1.5, -2.0))),
        check_exact=True,
    )


def test_each_geometry_skews_about_its_own_origin(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords).with_columns(
        origin=geo.mean_coordinate("polygon")
    )
    out = df.select(geo.skew("polygon", 30, -10, origin="origin"))

    for row, origin in enumerate(df["origin"].ext.storage().to_list()):
        alone = df.slice(row, 1).select(
            geo.skew("polygon", 30, -10, origin=(origin["x"], origin["y"]))
        )
        assert_frame_equal(out.slice(row, 1), alone, check_exact=True)


def test_skewing_about_its_own_middle_keeps_the_middle(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons(ring_coords)
    middle = geo.mean_coordinate("polygon")
    skewed = df.select(geo.skew("polygon", 30, -10, origin=middle))

    assert_frame_equal(
        coordinates(skewed.select(middle), "polygon"),
        coordinates(df.select(middle), "polygon"),
    )


def test_points_skew_about_a_single_origin(coords: pl.DataFrame) -> None:
    df = coords.select(XY.point())

    assert_frame_equal(
        df.select(geo.skew("point", 45, -45, origin=_origins(1.0, 2.0))),
        df.select(geo.skew("point", 45, -45, origin=(1.0, 2.0))),
        check_exact=True,
    )


def test_a_null_origin_gives_a_null_geometry(line_coords: pl.DataFrame) -> None:
    df = XY.lines(line_coords).with_columns(
        origin=pl.when(pl.int_range(pl.len()) == 0).then(geo.mean_coordinate("line"))
    )
    out = df.select(geo.skew("line", 30, origin="origin"))

    assert out["line"].is_null().to_list() == [False, True]


def test_an_origin_column_has_to_hold_points(line_coords: pl.DataFrame) -> None:
    lf = XY.lines(line_coords).lazy()

    with pytest.raises(TypeError, match="`origin` has to be a `geoarrow.point`"):
        lf.select(geo.skew("line", 30, origin="line")).collect_schema()


def test_the_namespace_forwards_an_origin_column(line_coords: pl.DataFrame) -> None:
    df = XY.lines(line_coords).with_columns(origin=geo.mean_coordinate("line"))

    assert_frame_equal(
        df.select(gpl.col("line").geo.skew(30, -10, origin="origin")),
        df.select(geo.skew("line", 30, -10, origin="origin")),
    )
