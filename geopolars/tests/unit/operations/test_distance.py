"""How far apart two points are: on the plane without a CRS, along the
ellipsoid with one."""

from __future__ import annotations

import math
from typing import Literal

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import PointType, PointXY
from tests.unit.conftest import XY, Dimension

WGS84 = "EPSG:4326"

# The example from the `geo` crate's `Geodesic` docs, in metres.
NEW_YORK = (-74.006, 40.7128)
LONDON = (-0.1278, 51.5074)
NEW_YORK_TO_LONDON = 5_585_234.0


def _pairs(
    a: list[tuple[float, float] | None],
    b: list[tuple[float, float] | None],
    crs: str | None = None,
) -> pl.DataFrame:
    """Two XY point columns, `a` and `b`, with a null point where it says `None`."""
    xy = pl.Struct({"x": pl.Float64, "y": pl.Float64})

    def column(points: list[tuple[float, float] | None]) -> pl.Series:
        rows = [None if p is None else {"x": p[0], "y": p[1]} for p in points]
        return pl.Series(rows, dtype=xy).ext.to(PointXY(crs=crs))

    return pl.DataFrame({"a": column(a), "b": column(b)})


def _distances(df: pl.DataFrame) -> list[float | None]:
    return df.select(geo.distance("a", "b")).to_series().to_list()


def test_without_a_crs_it_is_pythagoras() -> None:
    df = _pairs([(0.0, 0.0), (1.0, 1.0)], [(3.0, 4.0), (1.0, 1.0)])

    assert _distances(df) == [5.0, 0.0]


def test_negative_offsets_measure_the_same() -> None:
    df = _pairs([(0.0, 0.0), (-1.0, 2.0)], [(3.0, 4.0), (5.0, -6.0)])

    assert _distances(df) == [5.0, 10.0]


def test_without_a_crs_lon_lat_comes_out_in_degrees() -> None:
    """Naive on purpose: a degree of longitude is counted as long as one of
    latitude, wherever on the globe it is."""
    df = _pairs([NEW_YORK], [LONDON])

    (got,) = _distances(df)

    assert got == pytest.approx(math.hypot(73.8782, 10.7946))


def test_against_a_point_without_z_the_height_is_left_out(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = coords.select(dimension.point())
    origin = pl.select(geo.point(pl.lit(0.0), pl.lit(0.0)).alias("origin"))

    got = df.select(geo.distance("point", origin["origin"])).to_series().to_list()

    assert got == pytest.approx([math.hypot(1.0, 3.0), math.hypot(2.5, 4.5), 0.0])


def test_between_two_points_with_z_the_height_counts(
    coords: pl.DataFrame, dimension: Dimension
) -> None:
    """`m` is a measure, not a position, so it never counts."""
    df = coords.select(dimension.point())
    zeros = pl.DataFrame({axis: [0.0] for axis in ("x", "y", "z", "m")})
    origin = zeros.select(dimension.point())["point"]

    got = df.select(geo.distance("point", origin)).to_series().to_list()

    z = (10.0, 20.0, 0.0) if dimension.has_z else (0.0, 0.0, 0.0)
    expected = [
        math.hypot(1.0, 3.0, z[0]),
        math.hypot(2.5, 4.5, z[1]),
        0.0,
    ]
    assert got == pytest.approx(expected)


def _heights(
    a: tuple[float, float, float], b: tuple[float, float, float], crs: str | None
) -> pl.DataFrame:
    """One pair of XYZ points, `a` and `b`."""
    return pl.DataFrame({"i": [0]}).select(
        geo.point(*map(pl.lit, a), crs=crs).alias("a"),
        geo.point(*map(pl.lit, b), crs=crs).alias("b"),
    )


def test_without_a_crs_z_is_pythagoras_in_space() -> None:
    df = _heights((0.0, 0.0, 0.0), (3.0, 4.0, 12.0), crs=None)

    assert _distances(df) == [13.0]


def test_with_a_crs_straight_up_is_the_difference_in_height() -> None:
    df = _heights((*NEW_YORK, 10.0), (*NEW_YORK, 110.0), crs=WGS84)

    assert _distances(df) == pytest.approx([100.0])


# New York Long Island, in US survey feet.
NY_FEET = "EPSG:2263"


def test_with_a_crs_in_feet_the_distance_is_in_feet() -> None:
    # 3000 by 4000 feet on the grid; the ellipsoid differs from the grid
    # by the projection's scale factor, well under a part in ten thousand here.
    df = _pairs([(984_000.0, 200_000.0)], [(987_000.0, 204_000.0)], crs=NY_FEET)

    assert _distances(df) == pytest.approx([5000.0], rel=1e-4)


def test_with_a_crs_in_feet_the_height_is_in_feet() -> None:
    df = _heights((984_000.0, 200_000.0, 0.0), (984_000.0, 200_000.0, 1000.0), NY_FEET)

    assert _distances(df) == pytest.approx([1000.0])


def test_with_a_crs_in_feet_the_height_is_added_to_the_geodesic() -> None:
    a, b = (984_000.0, 200_000.0), (987_000.0, 204_000.0)
    df = _heights((*a, 0.0), (*b, 12_000.0), NY_FEET)
    (flat,) = _distances(_pairs([a], [b], crs=NY_FEET))

    (got,) = _distances(df)

    assert got == pytest.approx(math.hypot(flat, 12_000.0))


def test_with_a_crs_the_height_is_added_to_the_geodesic() -> None:
    climb = 10_000.0
    df = _heights((*NEW_YORK, 0.0), (*LONDON, climb), crs=WGS84)
    (flat,) = _distances(_pairs([NEW_YORK], [LONDON], crs=WGS84))

    (got,) = _distances(df)

    assert got == pytest.approx(math.hypot(flat, climb))


def test_the_order_of_the_points_does_not_matter() -> None:
    df = _pairs([(1.0, 2.0), NEW_YORK], [(4.0, 6.0), LONDON])

    forward = df.select(geo.distance("a", "b")).to_series()
    backward = df.select(geo.distance("b", "a").alias("a")).to_series()

    assert_series_equal(forward, backward)


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_a_null_point_has_no_distance(crs: str | None) -> None:
    df = _pairs([(0.0, 0.0), None, None], [None, (3.0, 4.0), None], crs=crs)

    assert _distances(df) == [None, None, None]


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_a_single_point_is_measured_against_every_row(crs: str | None) -> None:
    df = _pairs([(0.0, 0.0), (3.0, 4.0)], [(0.0, 0.0), (0.0, 0.0)], crs=crs)
    one = df["b"].head(1)

    expected = df.select(geo.distance("a", "b")).to_series()

    assert_series_equal(df.select(geo.distance("a", one)).to_series(), expected)
    swapped = df.select(geo.distance(one, "a").alias("a")).to_series()
    assert_series_equal(swapped, expected)


def test_with_a_crs_it_is_the_geodesic_in_meters() -> None:
    """On WGS 84, for EPSG:4326."""
    df = _pairs([NEW_YORK], [LONDON], crs=WGS84)

    (got,) = _distances(df)

    assert round(got) == NEW_YORK_TO_LONDON


def test_a_projected_crs_is_measured_on_its_own_ellipsoid() -> None:
    """The Dutch national grid is in metres already, but measured along the
    ellipsoid it gives the same distance as the same points in the lon/lat it
    projects from (Amersfoort, on Bessel 1841)."""
    amsterdam, utrecht = (4.9041, 52.3676), (5.1214, 52.0907)
    lonlat = _pairs([amsterdam], [utrecht], crs="EPSG:4289")
    rd = lonlat.select(geo.to_crs(pl.all(), "EPSG:28992"))

    (expected,) = _distances(lonlat)
    (got,) = _distances(rd)

    assert got == pytest.approx(expected, abs=1e-3)


def test_the_ellipsoid_is_the_one_the_crs_declares() -> None:
    """On a unit sphere, a quarter of the equator is a quarter turn long."""
    sphere = "+proj=longlat +R=1 +type=crs"
    df = _pairs([(0.0, 0.0), (0.0, 0.0)], [(90.0, 0.0), (0.0, 90.0)], crs=sphere)

    assert _distances(df) == pytest.approx([math.pi / 2, math.pi / 2])


def test_the_same_numbers_on_another_ellipsoid_are_another_distance() -> None:
    """Bessel 1841 is 740 m smaller than WGS 84 at the equator, which shortens
    the distance roughly in proportion. Only a sanity check on the size of the
    difference: the unit sphere above pins the ellipsoid down exactly."""
    wgs84 = _pairs([NEW_YORK], [LONDON], crs=WGS84)
    bessel = _pairs([NEW_YORK], [LONDON], crs="EPSG:4289")

    (on_wgs84,) = _distances(wgs84)
    (on_bessel,) = _distances(bessel)

    in_proportion = NEW_YORK_TO_LONDON * 739.845 / 6_378_137.0
    assert on_wgs84 - on_bessel == pytest.approx(in_proportion, rel=0.2)


def test_metadata_without_a_crs_is_still_planar() -> None:
    spherical = PointType.ext_from_params(
        "geoarrow.point", PointXY().ext_storage(), '{"edges":"spherical"}'
    )
    df = _pairs([(0.0, 0.0)], [(3.0, 4.0)]).select(
        pl.all().ext.storage().ext.to(spherical)
    )

    assert _distances(df) == [5.0]


def test_planar_ignores_the_crs() -> None:
    df = _pairs([(0.0, 0.0)], [(3.0, 4.0)], crs=WGS84)

    out = df.select(geo.distance("a", "b", method="planar")).to_series().to_list()

    assert out == [5.0]


def test_geodesic_without_a_crs_is_refused() -> None:
    lf = _pairs([NEW_YORK], [LONDON]).lazy()

    with pytest.raises(TypeError, match="neither point declares one"):
        lf.select(geo.distance("a", "b", method="geodesic")).collect_schema()


def test_points_in_different_crss_are_refused_while_resolving_the_schema() -> None:
    a = _pairs([NEW_YORK], [LONDON], crs=WGS84)
    lf = a.lazy().select(geo.distance("a", geo.to_crs("b", "EPSG:3857")))

    with pytest.raises(ComputeError, match="different CRSs"):
        lf.collect_schema()


def test_a_crs_on_only_one_point_is_refused_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame(
        {
            "a": _pairs([NEW_YORK], [LONDON], crs=WGS84)["a"],
            "b": _pairs([NEW_YORK], [LONDON])["b"],
        }
    ).select(geo.distance("a", "b"))

    with pytest.raises(ComputeError, match="only one of the"):
        lf.collect_schema()


def test_a_crs_not_on_longitude_latitude_is_refused_while_resolving_the_schema() -> (
    None
):
    """Earth-centred XYZ has no ellipsoid surface to measure a distance on."""
    lf = _pairs([NEW_YORK], [LONDON], crs="EPSG:4978").lazy()

    with pytest.raises(ComputeError, match="not defined on longitude/latitude"):
        lf.select(geo.distance("a", "b")).collect_schema()


def test_only_points_are_measured(line_coords: pl.DataFrame) -> None:
    lf = XY.lines(line_coords).lazy().select(geo.distance("line", "line"))

    with pytest.raises(TypeError, match="between two `geoarrow.point` columns"):
        lf.collect_schema()


def test_rejects_a_non_geometry_while_resolving_the_schema() -> None:
    lf = pl.LazyFrame({"a": [1.0], "b": [2.0]}).select(geo.distance("a", "b"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_the_result_is_named_after_the_first_point(crs: str | None) -> None:
    df = _pairs([(0.0, 0.0)], [(3.0, 4.0)], crs=crs)

    out = df.select(geo.distance("b", "a"))

    assert out.schema == pl.Schema({"b": pl.Float64})


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
@pytest.mark.parametrize("method", [None, "planar"])
def test_the_namespace_matches_the_function(
    crs: str | None, method: Literal["planar"] | None
) -> None:
    df = _pairs([NEW_YORK, (0.0, 0.0)], [LONDON, (3.0, 4.0)], crs=crs)

    assert_series_equal(
        df.select(gpl.col("a").geo.distance("b", method=method)).to_series(),
        df.select(geo.distance("a", "b", method=method)).to_series(),
    )
