"""`to_crs` on geometries with a `z`."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal

from geopolars import geo

WGS84 = "EPSG:4326"
WGS84_3D = "EPSG:4979"
GEOCENTRIC = "EPSG:4978"
WEB_MERCATOR = "EPSG:3857"
# New York Long Island in US survey feet, with NAVD88 heights in US survey feet.
NY_FEET = "EPSG:2263+6360"
# NAD83 longitude/latitude, with NAVD88 heights in metres.
NAD83_METRES = "EPSG:4269+5703"


def _points(crs: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [5.0, 5.0, None],
            "y": [52.0, 52.0, None],
            "z": [0.0, 1000.0, None],
            "m": [7.0, 8.0, None],
        }
    ).select(geo.point("x", "y", "z", "m", crs=crs).alias("point"))


def test_x_and_y_that_depend_on_height_get_it_from_z() -> None:
    out = _points(WGS84_3D).select(geo.to_crs("point", GEOCENTRIC))

    x, y, z, m = out["point"].ext.storage().struct.unnest().get_columns()
    assert x.to_list()[:2] == pytest.approx([3919986.754103, 3920600.072800])
    assert y.to_list()[:2] == pytest.approx([342954.402156, 343008.060589])
    assert z.to_list()[:2] == pytest.approx([5002803.345483, 5003591.356236])
    assert_series_equal(m, pl.Series("m", [7.0, 8.0, None]), check_names=False)
    assert out["point"].null_count() == 1


def test_z_is_converted_to_the_unit_of_the_target_crs() -> None:
    lines = pl.DataFrame(
        {"x": [[1e6, 1e6]], "y": [[2e5, 2e5]], "z": [[100.0, 0.0]]}
    ).select(geo.line_string("x", "y", "z", crs=NY_FEET).alias("line"))

    out = lines.select(geo.z(geo.to_crs("line", NAD83_METRES)))

    assert out["line"].to_list() == [pytest.approx([30.480061, 0.0])]


@pytest.mark.parametrize(
    ("source", "target", "flag"),
    [
        (WGS84_3D, WEB_MERCATOR, "allow_projection_with_unscaled_z"),
        (WGS84, WEB_MERCATOR, "allow_projection_with_unscaled_z"),
        (WGS84, WGS84_3D, "assume_z_in_target_crs"),
    ],
)
def test_a_z_that_one_crs_does_not_specify_needs_a_flag(
    source: str, target: str, flag: str
) -> None:
    points = _points(source)

    with pytest.raises(ComputeError, match=flag):
        points.lazy().select(geo.to_crs("point", target)).collect_schema()

    out = points.select(geo.to_crs("point", target, **{flag: True}))
    assert_series_equal(
        out.select(geo.z("point"))["point"], points.select(geo.z("point"))["point"]
    )


def test_a_geometry_without_z_ignores_the_flags() -> None:
    points = pl.DataFrame({"x": [5.0], "y": [52.0]}).select(
        geo.point("x", "y", crs=WGS84).alias("point")
    )

    plain = points.select(geo.to_crs("point", WEB_MERCATOR))
    flagged = points.select(
        geo.to_crs(
            "point",
            WEB_MERCATOR,
            allow_projection_with_unscaled_z=True,
            assume_z_in_target_crs=True,
        )
    )

    assert plain.equals(flagged)


def test_a_bound_crs_reprojects_a_geometry_without_z() -> None:
    # A BoundCRS, of which PROJ cannot tell whether it specifies z.
    bessel = "+proj=longlat +ellps=bessel +towgs84=565.4,50.3,465.6 +no_defs +type=crs"
    points = pl.DataFrame({"x": [5.0], "y": [52.0]}).select(
        geo.point("x", "y", crs=bessel).alias("point")
    )

    out = points.select(geo.to_crs("point", WEB_MERCATOR))

    # (5, 52) in Web Mercator, give or take the datum shift.
    xy = out.select(geo.x("point"), geo.y("point").alias("y")).row(0)
    assert xy == pytest.approx((556597.5, 6800125.5), abs=200)
