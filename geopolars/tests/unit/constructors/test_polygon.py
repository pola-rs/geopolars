"""Building `geoarrow.polygon` columns with the `geopolars.geo.polygons_from_*` constructors."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import (
    MultiLineStringXY,
    PolygonXY,
)
from tests.unit.conftest import COORDS, Dimension, ring_coordinates

# A list of bare coordinate structs: what a ring stores, before it is one.
_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

# A list of those: what a polygon stores, before it is one.
_XY_RINGS = pl.List(_XY_VERTICES)

# One coordinate column per axis: one list per polygon, its exterior ring.
_XY_COORDS = {"lon": pl.List(pl.Float64), "lat": pl.List(pl.Float64)}

# One coordinate column per axis: one list per polygon, of one list per ring.
_XY_NESTED_COORDS = {
    "lon": pl.List(pl.List(pl.Float64)),
    "lat": pl.List(pl.List(pl.Float64)),
}

# One closed triangular ring, the smallest polygon that is a polygon.
_TRIANGLE = [
    [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.0},
        {"x": 0.0, "y": 1.0},
        {"x": 0.0, "y": 0.0},
    ]
]


def test_the_rings_decide_the_dtype(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """A polygon has the dimension of the rings it is built from."""
    df = dimension.polygons(ring_coords)

    assert df.schema["polygon"] == dimension.polygon_dtype()


def test_accepts_bare_lists_of_coordinate_structs() -> None:
    """The rings do not have to be linestrings already:
    a list of the vertex lists a linestring wraps is the same storage,
    and is read the same way."""
    df = pl.DataFrame({"rings": [_TRIANGLE]}, schema={"rings": _XY_RINGS}).select(
        geo.polygons_from_rings_in_single_column("rings").alias("polygon")
    )

    assert df.schema["polygon"] == PolygonXY()
    assert_frame_equal(
        ring_coordinates(df),
        pl.DataFrame({"x": [0.0, 1.0, 0.0, 0.0], "y": [0.0, 0.0, 1.0, 0.0]}),
    )


def test_rings_keep_their_order() -> None:
    """Which ring is which is positional:
    the first is the exterior ring and the rest are holes,
    so reordering them is another polygon."""
    hole = [
        {"x": 0.2, "y": 0.2},
        {"x": 0.4, "y": 0.2},
        {"x": 0.2, "y": 0.4},
        {"x": 0.2, "y": 0.2},
    ]
    df = pl.DataFrame(
        {"rings": [[*_TRIANGLE, hole]]}, schema={"rings": _XY_RINGS}
    ).select(geo.polygons_from_rings_in_single_column("rings").alias("polygon"))
    swapped = df.select(
        pl.col("polygon").ext.storage().list.reverse().alias("rings")
    ).select(geo.polygons_from_rings_in_single_column("rings").alias("polygon"))

    assert_frame_equal(
        ring_coordinates(swapped),
        pl.DataFrame(
            {
                "x": [0.2, 0.4, 0.2, 0.2, 0.0, 1.0, 0.0, 0.0],
                "y": [0.2, 0.2, 0.4, 0.2, 0.0, 0.0, 1.0, 0.0],
            }
        ),
    )


def test_an_empty_list_is_an_empty_polygon() -> None:
    """A polygon with no rings is a geometry this layout can hold,
    and is not a null polygon.
    See https://geoarrow.org/format.html#empty-geometries"""
    df = pl.DataFrame({"rings": [[]]}, schema={"rings": _XY_RINGS}).select(
        geo.polygons_from_rings_in_single_column("rings").alias("polygon")
    )

    assert df.schema["polygon"] == PolygonXY()
    assert df["polygon"].is_null().to_list() == [False]
    assert df.select(pl.col("polygon").ext.storage().list.len())[
        "polygon"
    ].to_list() == [0]


def test_a_null_ring_nulls_the_whole_polygon() -> None:
    """GeoArrow allows nulls only at the outermost level,
    so a polygon with a null ring is a null polygon.
    See https://geoarrow.org/format.html#missing-values-null"""
    df = pl.DataFrame(
        {"rings": [[*_TRIANGLE, None], _TRIANGLE, None]},
        schema={"rings": _XY_RINGS},
    ).select(geo.polygons_from_rings_in_single_column("rings").alias("polygon"))

    assert df["polygon"].is_null().to_list() == [True, False, True]


def test_a_null_vertex_nulls_the_whole_polygon() -> None:
    """The same rule one level further down than for a linestring:
    a ring with a null vertex cannot stand,
    and neither can the polygon around it."""
    df = pl.DataFrame(
        {"rings": [[[{"x": 1.0, "y": 2.0}, None]], _TRIANGLE]},
        schema={"rings": _XY_RINGS},
    ).select(geo.polygons_from_rings_in_single_column("rings").alias("polygon"))

    assert df["polygon"].is_null().to_list() == [True, False]


def test_a_null_coordinate_nulls_the_whole_polygon() -> None:
    """And one level down again:
    a vertex that is present but has no `y` is not a vertex,
    so its ring cannot stand, so its polygon cannot either."""
    df = pl.DataFrame(
        {"rings": [[[{"x": 1.0, "y": None}]], _TRIANGLE]},
        schema={"rings": _XY_RINGS},
    ).select(geo.polygons_from_rings_in_single_column("rings").alias("polygon"))

    assert df["polygon"].is_null().to_list() == [True, False]


def test_rejects_rings_that_are_not_coordinates() -> None:
    df = pl.DataFrame({"rings": [[[1.0, 2.0]]]})

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geo.polygons_from_rings_in_single_column("rings"))


def test_rejects_a_column_that_is_not_a_list() -> None:
    """A linestring column is a ring per row, not a polygon per row."""
    df = pl.DataFrame(
        {"vertices": _TRIANGLE}, schema={"vertices": _XY_VERTICES}
    ).select(geo.line_string("vertices").alias("line"))

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geo.polygons_from_rings_in_single_column("line"))


def test_rejects_a_list_of_vertices() -> None:
    """a list of coordinates is a ring, not a list of rings."""
    df = pl.DataFrame({"vertices": _TRIANGLE}, schema={"vertices": _XY_VERTICES})

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geo.polygons_from_rings_in_single_column("vertices"))


def test_rejects_a_list_of_points() -> None:
    """A polygon is built out of rings, and a point is not a ring."""
    df = (
        pl.DataFrame({"x": [1.0], "y": [2.0]})
        .select(geo.point("x", "y").alias("point"))
        .select(pl.col("point").implode())
    )

    with pytest.raises(ComputeError, match="expected a list of"):
        df.select(geo.polygons_from_rings_in_single_column("point"))


def test_empty_frame_keeps_its_dtype() -> None:
    """A zero-row build still produces a polygon column, not a bare list."""
    df = pl.DataFrame(schema={"rings": _XY_RINGS}).select(
        geo.polygons_from_rings_in_single_column("rings").alias("polygon")
    )

    assert df.height == 0
    assert df.schema["polygon"] == PolygonXY()


def test_rejects_bad_rings_while_resolving_the_schema() -> None:
    """The dimension is read off the input's dtype,
    so rings that are not coordinates are a schema error."""
    lf = pl.LazyFrame({"plot": ["north"]}).select(
        geo.polygons_from_rings_in_single_column("plot")
    )

    with pytest.raises(ComputeError, match="expected a list of"):
        lf.collect_schema()


def test_coordinate_columns_decide_the_dtype(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.polygons_from_coords(ring_coords)

    assert df.schema["polygon"] == dimension.polygon_dtype()


def test_coordinate_columns_build_the_same_polygons(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The two forms are two ways to spell one polygon, not two geometries."""
    exteriors = ring_coords.filter(pl.col("ring") == 0)
    assert_frame_equal(
        dimension.polygons_from_coords(ring_coords), dimension.polygons(exteriors)
    )


def test_a_null_coordinate_nulls_the_whole_polygon_of_coords() -> None:
    """A vertex with a null `y` cannot stand, so neither can the ring around
    it, nor the polygon around that."""
    df = pl.DataFrame(
        {
            "lon": [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "lat": [[0.0, None, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        },
        schema=_XY_COORDS,
    ).select(geo.polygons_from_coordinates("lon", "lat").alias("polygon"))

    assert df["polygon"].is_null().to_list() == [True, False]


def test_rejects_coordinate_columns_that_nest_differently() -> None:
    df = pl.DataFrame({"lon": [[1.0, 3.0]], "lat": [[2.0]]}, schema=_XY_COORDS)

    with pytest.raises(ComputeError, match="do not nest the same way"):
        df.select(geo.polygons_from_coordinates("lon", "lat"))


def test_rejects_coordinate_columns_of_rings() -> None:
    """One list per ring is how holes would be spelled, which coordinate columns cannot."""
    lf = pl.LazyFrame({"lon": [[[1.0]]], "lat": [[[2.0]]]}).select(
        geo.polygons_from_coordinates("lon", "lat")
    )

    with pytest.raises(ComputeError, match="to hold lists of f64, got"):
        lf.collect_schema()


def test_empty_frame_of_coordinates_keeps_its_dtype() -> None:
    df = pl.DataFrame(schema=_XY_COORDS).select(
        geo.polygons_from_coordinates("lon", "lat").alias("polygon")
    )

    assert df.height == 0
    assert df.schema["polygon"] == PolygonXY()


def test_nested_coordinate_columns_build_the_same_polygons(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """Holes included: grouping twice is what nests them one list deeper."""
    nested = (
        ring_coords.group_by("polygon", "ring", maintain_order=True)
        .agg(COORDS)
        .group_by("polygon", maintain_order=True)
        .agg(COORDS)
        .select(
            dimension.of_coords(geo.polygons_from_nested_coordinates).alias("polygon")
        )
    )

    assert_frame_equal(nested, dimension.polygons(ring_coords))


def test_nested_coordinate_columns_keep_their_rings_in_order() -> None:
    """The nesting is the ring structure: the first list is the exterior ring."""
    df = pl.DataFrame(
        {
            "lon": [[[0.0, 1.0, 0.0, 0.0], [0.2, 0.4, 0.2, 0.2]]],
            "lat": [[[0.0, 0.0, 1.0, 0.0], [0.2, 0.2, 0.4, 0.2]]],
        }
    ).select(geo.polygons_from_nested_coordinates("lon", "lat").alias("polygon"))

    assert df.select(pl.col("polygon").ext.storage().list.len())[
        "polygon"
    ].to_list() == [2]
    assert_frame_equal(
        ring_coordinates(df),
        pl.DataFrame(
            {
                "x": [0.0, 1.0, 0.0, 0.0, 0.2, 0.4, 0.2, 0.2],
                "y": [0.0, 0.0, 1.0, 0.0, 0.2, 0.2, 0.4, 0.2],
            }
        ),
    )


def test_a_null_coordinate_nulls_the_whole_polygon_of_nested_coords() -> None:
    df = pl.DataFrame(
        {
            "lon": [[[0.0, 1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]],
            "lat": [[[0.0, None, 1.0, 0.0]], [[0.0, 0.0, 1.0, 0.0]]],
        },
        schema=_XY_NESTED_COORDS,
    ).select(geo.polygons_from_nested_coordinates("lon", "lat").alias("polygon"))

    assert df["polygon"].is_null().to_list() == [True, False]


@pytest.mark.parametrize(
    ("lon", "lat"),
    [
        # A ring's vertex counts disagree.
        ([[[1.0, 3.0]]], [[[2.0]]]),
        # The polygons disagree about how many rings they have.
        ([[[1.0], [3.0]]], [[[2.0]]]),
    ],
)
def test_rejects_nested_coordinate_columns_that_nest_differently(
    lon: list[list[list[float]]], lat: list[list[list[float]]]
) -> None:
    df = pl.DataFrame({"lon": lon, "lat": lat}, schema=_XY_NESTED_COORDS)

    with pytest.raises(ComputeError, match="do not nest the same way"):
        df.select(geo.polygons_from_nested_coordinates("lon", "lat"))


def test_rejects_nested_coordinate_columns_nested_only_once() -> None:
    lf = pl.LazyFrame({"lon": [[1.0]], "lat": [[2.0]]}).select(
        geo.polygons_from_nested_coordinates("lon", "lat")
    )

    with pytest.raises(ComputeError, match="lists of lists of f64, one per polygon"):
        lf.collect_schema()


def test_nested_coordinate_columns_are_checked_and_closed() -> None:
    df = pl.DataFrame(
        {
            "lon": [[[0.0, 1.0, 0.0]], [[0.0, 1.0]]],
            "lat": [[[0.0, 0.0, 1.0]], [[0.0, 0.0]]],
        },
        schema=_XY_NESTED_COORDS,
    )

    checked = df.select(
        geo.polygons_from_nested_coordinates("lon", "lat").alias("polygon")
    )
    closed = df.select(
        geo.polygons_from_nested_coordinates(
            "lon", "lat", close=geo.CloseRings()
        ).alias("polygon")
    )

    assert checked["polygon"].is_null().to_list() == [True, True]
    assert closed["polygon"].ext.storage().to_list() == [_TRIANGLE, None]


def _ring_xy(*xy: tuple[float, float]) -> list[dict[str, float]]:
    return [{"x": x, "y": y} for x, y in xy]


# The triangle again, without the vertex closing it.
_OPEN = _ring_xy((0, 0), (1, 0), (0, 1))


def _from_rings(
    *polygons: list | None, close: geo.CloseRings | None = None
) -> pl.Series:
    return (
        pl.DataFrame({"rings": list(polygons)}, schema={"rings": _XY_RINGS})
        .select(
            geo.polygons_from_rings_in_single_column("rings", close=close).alias(
                "polygon"
            )
        )
        .to_series()
    )


def test_a_ring_that_is_no_ring_nulls_the_polygon() -> None:
    """Open, or closed with too few vertices to enclose anything,
    in the exterior or in a hole."""
    there_and_back = _ring_xy((0, 0), (1, 0), (0, 0))
    out = _from_rings(
        _TRIANGLE, [_OPEN], [there_and_back], [*_TRIANGLE, _OPEN], [*_TRIANGLE, []], []
    )

    assert out.is_null().to_list() == [False, True, True, True, True, False]


def test_close_closes_the_open_rings() -> None:
    out = _from_rings([_OPEN], [*_TRIANGLE, _OPEN], close=geo.CloseRings())

    assert out.is_null().to_list() == [False, False]
    assert out.ext.storage().to_list() == [_TRIANGLE, [*_TRIANGLE, *_TRIANGLE]]


def test_close_closes_a_ring_that_is_almost_closed() -> None:
    almost = _ring_xy((1e6, 0), (2e6, 0), (2e6, 1), (1e6 + 1e-4, 0))

    closed = _from_rings([almost], close=geo.CloseRings()).ext.storage().to_list()
    assert closed == [[[*almost, almost[0]]]]


def test_close_cannot_save_a_ring_that_is_too_short() -> None:
    """A polygon of only an empty ring is the empty polygon, so it is kept."""
    out = _from_rings([_ring_xy((0, 0), (1, 0))], [[]], close=geo.CloseRings())

    assert out.is_null().to_list() == [True, False]


def test_close_on_a_sliced_column() -> None:
    df = pl.DataFrame(
        {"rings": [[_OPEN], [_OPEN], _TRIANGLE]}, schema={"rings": _XY_RINGS}
    ).slice(1, 2)

    out = df.select(
        geo.polygons_from_rings_in_single_column("rings", close=geo.CloseRings()).alias(
            "polygon"
        )
    )

    assert out["polygon"].ext.storage().to_list() == [_TRIANGLE, _TRIANGLE]


def test_coordinate_columns_are_checked_and_closed_too() -> None:
    df = pl.DataFrame(
        {"lon": [[0.0, 1.0, 0.0], [0.0, 1.0]], "lat": [[0.0, 0.0, 1.0], [0.0, 0.0]]},
        schema=_XY_COORDS,
    )

    checked = df.select(geo.polygons_from_coordinates("lon", "lat").alias("polygon"))
    closed = df.select(
        geo.polygons_from_coordinates("lon", "lat", close=geo.CloseRings()).alias(
            "polygon"
        )
    )

    assert checked["polygon"].is_null().to_list() == [True, True]
    assert closed["polygon"].ext.storage().to_list() == [_TRIANGLE, None]


# A hole inside `_TRIANGLE`.
_HOLE = _ring_xy((0.2, 0.2), (0.4, 0.2), (0.2, 0.4), (0.2, 0.2))
_EXTERIOR = _TRIANGLE[0]


def _from_separate_rings(
    exteriors: list,
    holes: list | None = None,
    holes_dtype: pl.DataType = _XY_RINGS,
    close: geo.CloseRings | None = None,
) -> pl.Series:
    columns = {"exterior": exteriors}
    schema = {"exterior": _XY_VERTICES}
    if holes is not None:
        columns["holes"] = holes
        schema["holes"] = holes_dtype
    return (
        pl.DataFrame(columns, schema=schema)
        .select(
            geo.polygons_from_rings_in_separate_columns(
                "exterior", "holes" if holes is not None else None, close=close
            ).alias("polygon")
        )
        .to_series()
    )


def test_separate_columns_take_a_list_of_holes() -> None:
    out = _from_separate_rings([_EXTERIOR, _EXTERIOR], [[_HOLE, _HOLE], []])

    assert out.dtype == PolygonXY()
    assert out.ext.storage().to_list() == [
        [_EXTERIOR, _HOLE, _HOLE],
        [_EXTERIOR],
    ]


def test_separate_columns_take_a_multilinestring_of_holes() -> None:
    exteriors = pl.Series("exterior", [_EXTERIOR, _EXTERIOR], dtype=_XY_VERTICES)
    holes = pl.Series("holes", [[_HOLE, _HOLE], []], dtype=_XY_RINGS)
    df = pl.DataFrame([exteriors, holes.ext.to(MultiLineStringXY())])

    out = df.select(
        geo.polygons_from_rings_in_separate_columns("exterior", "holes")
    ).to_series()

    assert out.ext.storage().to_list() == [
        [_EXTERIOR, _HOLE, _HOLE],
        [_EXTERIOR],
    ]


def test_separate_columns_take_a_single_hole() -> None:
    out = _from_separate_rings([_EXTERIOR], [_HOLE], holes_dtype=_XY_VERTICES)

    assert out.ext.storage().to_list() == [[_EXTERIOR, _HOLE]]


def test_separate_columns_take_linestrings() -> None:
    df = pl.DataFrame(
        {"exterior": [_EXTERIOR], "hole": [_HOLE]},
        schema={"exterior": _XY_VERTICES, "hole": _XY_VERTICES},
    ).select(
        geo.line_string("exterior", crs="EPSG:4326"),
        geo.line_string("hole", crs="EPSG:4326"),
    )
    single = df.select(
        geo.polygons_from_rings_in_separate_columns("exterior", "hole").alias("polygon")
    )
    listed = df.select(
        geo.polygons_from_rings_in_separate_columns(
            "exterior", pl.col("hole").implode()
        ).alias("polygon")
    )

    assert single.schema["polygon"] == PolygonXY(crs="EPSG:4326")
    assert_frame_equal(single, listed)
    assert single["polygon"].ext.storage().to_list() == [[_EXTERIOR, _HOLE]]


def _bare_exterior_and_hole() -> pl.DataFrame:
    return pl.DataFrame(
        {"exterior": [_EXTERIOR], "hole": [_HOLE]},
        schema={"exterior": _XY_VERTICES, "hole": _XY_VERTICES},
    )


def test_a_bare_exterior_takes_the_crs_of_the_holes() -> None:
    df = _bare_exterior_and_hole().with_columns(
        geo.line_string("hole", crs="EPSG:4326")
    )

    out = df.select(
        geo.polygons_from_rings_in_separate_columns("exterior", "hole")
    ).to_series()

    assert out.dtype == PolygonXY(crs="EPSG:4326")
    assert out.ext.storage().to_list() == [[_EXTERIOR, _HOLE]]


@pytest.mark.parametrize(
    ("exterior_crs", "hole_crs", "match"),
    [
        ("EPSG:4326", "EPSG:28992", "different CRSs"),
        ("EPSG:4326", None, "only one"),
        (None, "EPSG:4326", "only one"),
    ],
)
def test_separate_columns_refuse_holes_in_another_crs(
    exterior_crs: str | None, hole_crs: str | None, match: str
) -> None:
    lf = (
        _bare_exterior_and_hole()
        .lazy()
        .select(
            geo.line_string("exterior", crs=exterior_crs),
            geo.line_string("hole", crs=hole_crs),
        )
        .select(geo.polygons_from_rings_in_separate_columns("exterior", "hole"))
    )

    with pytest.raises(ComputeError, match=match):
        lf.collect_schema()


def test_holes_in_the_crs_passed_for_the_exterior_are_accepted() -> None:
    df = _bare_exterior_and_hole().select(
        geo.line_string("exterior"), geo.line_string("hole", crs="EPSG:4326")
    )

    out = df.select(
        geo.polygons_from_rings_in_separate_columns("exterior", "hole", crs="EPSG:4326")
    ).to_series()

    assert out.dtype == PolygonXY(crs="EPSG:4326")
    assert out.ext.storage().to_list() == [[_EXTERIOR, _HOLE]]


@pytest.mark.parametrize(
    "exterior",
    [
        pl.col("exterior"),
        geo.line_string("exterior"),
        geo.line_string("exterior", crs="EPSG:4326"),
    ],
    ids=["bare", "linestring", "linestring in the crs"],
)
def test_the_crs_passed_is_set_on_holes_without_one(exterior: pl.Expr) -> None:
    df = _bare_exterior_and_hole().with_columns(geo.line_string("hole"))

    out = df.select(
        geo.polygons_from_rings_in_separate_columns(exterior, "hole", crs="EPSG:4326")
    ).to_series()

    assert out.dtype == PolygonXY(crs="EPSG:4326")
    assert out.ext.storage().to_list() == [[_EXTERIOR, _HOLE]]


def test_holes_in_another_crs_than_the_one_passed_are_refused() -> None:
    lf = (
        _bare_exterior_and_hole()
        .lazy()
        .with_columns(geo.line_string("hole", crs="EPSG:4326"))
        .select(
            geo.polygons_from_rings_in_separate_columns(
                "exterior", "hole", crs="EPSG:28992"
            )
        )
    )

    with pytest.raises(ComputeError, match="already declare the CRS"):
        lf.collect_schema()


def test_separate_columns_without_holes_build_the_exterior_alone() -> None:
    out = _from_separate_rings([_EXTERIOR])

    assert out.to_frame("polygon").equals(_from_rings(_TRIANGLE).to_frame("polygon"))


def test_null_holes_are_no_holes() -> None:
    listed = _from_separate_rings([_EXTERIOR], [None])
    single = _from_separate_rings([_EXTERIOR], [None], holes_dtype=_XY_VERTICES)

    assert listed.ext.storage().to_list() == [[_EXTERIOR]]
    assert single.ext.storage().to_list() == [[_EXTERIOR]]


def test_a_null_exterior_is_a_null_polygon() -> None:
    out = _from_separate_rings([None, _EXTERIOR], [[_HOLE], [_HOLE]])

    assert out.is_null().to_list() == [True, False]


def test_a_null_ring_among_the_holes_nulls_the_polygon() -> None:
    out = _from_separate_rings([_EXTERIOR, _EXTERIOR], [[_HOLE, None], [_HOLE]])

    assert out.is_null().to_list() == [True, False]


def test_a_null_vertex_in_a_hole_nulls_the_polygon() -> None:
    broken = [*_HOLE[:3], None]
    listed = _from_separate_rings([_EXTERIOR], [[broken]])
    single = _from_separate_rings([_EXTERIOR], [broken], holes_dtype=_XY_VERTICES)

    assert listed.is_null().to_list() == [True]
    assert single.is_null().to_list() == [True]


def test_separate_columns_reject_holes_of_another_dimension() -> None:
    lf = pl.LazyFrame(
        {"exterior": [_EXTERIOR], "holes": [[]]},
        schema={
            "exterior": _XY_VERTICES,
            "holes": pl.List(
                pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64}))
            ),
        },
    ).select(geo.polygons_from_rings_in_separate_columns("exterior", "holes"))

    with pytest.raises(ComputeError, match="the exterior is xy but the holes are xyz"):
        lf.collect_schema()


def test_separate_columns_close_open_holes() -> None:
    checked = _from_separate_rings([_EXTERIOR], [[_OPEN]])
    closed = _from_separate_rings([_EXTERIOR], [[_OPEN]], close=geo.CloseRings())

    assert checked.is_null().to_list() == [True]
    assert closed.ext.storage().to_list() == [[_EXTERIOR, _EXTERIOR]]


def test_separate_columns_on_sliced_columns() -> None:
    df = pl.DataFrame(
        {
            "exterior": [_EXTERIOR, _EXTERIOR, _EXTERIOR],
            "holes": [[_HOLE], None, [_HOLE, _HOLE]],
        },
        schema={"exterior": _XY_VERTICES, "holes": _XY_RINGS},
    ).slice(1, 2)

    out = df.select(
        geo.polygons_from_rings_in_separate_columns("exterior", "holes").alias(
            "polygon"
        )
    )

    assert out["polygon"].ext.storage().to_list() == [
        [_EXTERIOR],
        [_EXTERIOR, _HOLE, _HOLE],
    ]
