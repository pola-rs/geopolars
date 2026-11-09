"""Whether two rings overlap: whether the areas they enclose share area."""

from __future__ import annotations

import random

import polars as pl
import pytest
import shapely
from polars.exceptions import ComputeError

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import LineStringXY, LineStringXYZ
from tests.unit.conftest import XY, Dimension

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

Ring = list[dict[str, float]]


def _ring(*xy: tuple[float, float]) -> Ring:
    """The ring through `xy`, closed back at the first."""
    return [{"x": x, "y": y} for x, y in [*xy, xy[0]]]


def _square(x: float, y: float, size: float = 1) -> Ring:
    return _ring((x, y), (x + size, y), (x + size, y + size), (x, y + size))


def _rings(name: str, rings: list[Ring | None]) -> pl.Series:
    return pl.Series(name, rings, dtype=_XY_VERTICES).ext.to(LineStringXY())


def _overlap(
    a: list[Ring | None], b: list[Ring | None], *, include_touching: bool = False
) -> list[bool | None]:
    df = pl.DataFrame([_rings("a", a), _rings("b", b)])
    out = df.select(geo.rings_overlap("a", "b", include_touching=include_touching))
    return out["a"].to_list()


def test_disjoint_rings_do_not() -> None:
    assert _overlap([_square(0, 0)], [_square(5, 5)]) == [False]


def test_a_partial_overlap_does() -> None:
    assert _overlap([_square(0, 0, 2)], [_square(1, 1, 2)]) == [True]


def test_one_inside_the_other_does() -> None:
    outer, inner = _square(0, 0, 4), _square(1, 1)
    assert _overlap([outer, inner], [inner, outer]) == [True, True]


def test_equal_rings_do() -> None:
    assert _overlap([_square(0, 0)], [_square(0, 0)]) == [True]


def test_the_direction_and_start_do_not_matter() -> None:
    counter = _square(0, 0, 2)
    clockwise = _ring((1, 1), (1, 3), (3, 3), (3, 1))
    assert _overlap([counter], [clockwise]) == [True]


def test_touching_only_counts_when_asked() -> None:
    corner, edge, far = _square(1, 1), _square(1, 0), _square(5, 5)
    a = [_square(0, 0)] * 3
    b = [corner, edge, far]

    assert _overlap(a, b) == [False, False, False]
    assert _overlap(a, b, include_touching=True) == [True, True, False]


def test_a_ring_inside_touching_from_within_overlaps_either_way() -> None:
    outer, inner = _square(0, 0, 4), _square(0, 0)
    assert _overlap([outer], [inner]) == [True]
    assert _overlap([outer], [inner], include_touching=True) == [True]


def test_null_stays_null() -> None:
    assert _overlap([None, _square(0, 0)], [_square(0, 0), None]) == [None, None]


def test_a_not_finite_vertex_gives_null() -> None:
    nan, inf = float("nan"), float("inf")
    broken = [_ring((0, 0), (nan, 0), (1, 1)), _ring((0, 0), (inf, 0), (1, 1))]
    assert _overlap(broken, [_square(0, 0)] * 2) == [None] * 2


@pytest.mark.parametrize(
    "line",
    [
        [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        _ring((0, 0), (1, 1)),
        [],
    ],
    ids=["not closed", "too few vertices", "empty"],
)
def test_anything_but_a_ring_raises(line: Ring) -> None:
    with pytest.raises(ComputeError, match="rings_overlap expects rings"):
        _overlap([line], [_square(0, 0)])
    with pytest.raises(ComputeError, match="rings_overlap expects rings"):
        _overlap([_square(0, 0)], [line])


def test_ignore_per_row_errors_gives_a_null_result() -> None:
    open_line = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    df = pl.DataFrame(
        [
            _rings("a", [open_line, _square(0, 0)]),
            _rings("b", [_square(0, 0), _square(0, 0)]),
        ]
    )
    out = df.select(geo.rings_overlap("a", "b", ignore_per_row_errors=True))
    assert out["a"].to_list() == [None, True]


@pytest.mark.parametrize("single_first", [True, False])
def test_ignore_per_row_errors_on_a_single_ring(single_first: bool) -> None:
    open_line = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    many = _rings("many", [_square(0, 0), open_line])
    good, bad = _rings("one", [_square(0, 0)]), _rings("one", [open_line])

    def overlap(one: pl.Series) -> list[bool | None]:
        df = pl.DataFrame(many)
        a, b = (one, "many") if single_first else ("many", one)
        out = df.select(geo.rings_overlap(a, b, ignore_per_row_errors=True))
        return out.to_series().to_list()

    assert overlap(good) == [True, None]
    assert overlap(bad) == [None, None]


def test_the_ends_have_to_be_exactly_equal() -> None:
    almost = _square(1e6, 0)
    almost[-1] = {"x": 1e6 + 1e-4, "y": 0.0}
    with pytest.raises(ComputeError, match="rings_overlap expects rings"):
        _overlap([almost], [_square(1e6, 0)])


def test_z_does_not_count() -> None:
    """Rings overlap in x and y at any height."""

    def at_height(ring: Ring, z: float) -> Ring:
        return [{**vertex, "z": z} for vertex in ring]

    dtype = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64}))
    df = pl.DataFrame(
        [
            pl.Series("a", [at_height(_square(0, 0, 2), 0)], dtype=dtype),
            pl.Series("b", [at_height(_square(1, 1, 2), 100)], dtype=dtype),
        ]
    ).select(pl.all().ext.to(LineStringXYZ()))

    assert df.select(geo.rings_overlap("a", "b"))["a"].to_list() == [True]


def test_every_dimension(dimension: Dimension) -> None:
    def ring(corners: list[tuple[float, float]]) -> Ring:
        return [
            {
                axis: {"x": float(x), "y": float(y)}.get(axis, 7.0)
                for axis in dimension.coords
            }
            for x, y in [*corners, corners[0]]
        ]

    square = ring([(0, 0), (2, 0), (2, 2), (0, 2)])
    shifted = ring([(1, 1), (3, 1), (3, 3), (1, 3)])
    far = ring([(5, 5), (6, 5), (6, 6), (5, 6)])
    df = pl.DataFrame(
        [pl.Series("a", [square, square]), pl.Series("b", [shifted, far])]
    ).select(pl.all().ext.to(dimension.linestring_dtype()))

    assert df.select(geo.rings_overlap("a", "b"))["a"].to_list() == [True, False]


def test_a_single_ring_is_compared_against_every_row() -> None:
    df = pl.DataFrame(_rings("a", [_square(0, 0), _square(5, 5), None]))
    hole = _rings("hole", [_square(0.5, 0.5, 0.25)])

    assert df.select(geo.rings_overlap("a", hole))["a"].to_list() == [
        True,
        False,
        None,
    ]


def test_sliced_columns_read_only_their_own_rows() -> None:
    df = pl.DataFrame(
        [
            _rings("a", [_square(0, 0), _square(0, 0), _square(5, 5)]),
            _rings("b", [_square(0, 0), _square(9, 9), _square(5, 5)]),
        ]
    ).slice(1, 2)

    assert df.select(geo.rings_overlap("a", "b"))["a"].to_list() == [False, True]


def _chunked(series: list[pl.Series]) -> pl.Series:
    out = pl.concat(series, rechunk=False)
    assert out.n_chunks() == len(series)
    return out


def test_columns_chunked_differently_are_compared_row_by_row() -> None:
    a = _chunked([_rings("a", [_square(0, 0)]), _rings("a", [_square(5, 5), None])])
    b = _chunked(
        [_rings("b", [_square(0, 0), _square(9, 9)]), _rings("b", [_square(0, 0)])]
    )

    out = pl.DataFrame([a, b]).select(geo.rings_overlap("a", "b"))

    assert out["a"].to_list() == [True, False, None]


def test_a_chunked_column_is_compared_against_a_single_ring() -> None:
    many = _chunked(
        [
            _rings("many", [_square(0, 0), _square(5, 5)]),
            _rings("many", [None, _square(0, 0)]),
        ]
    )
    hole = _rings("hole", [_square(0.5, 0.5, 0.25)])
    df = pl.DataFrame(many)
    expected = [True, False, None, True]

    assert df.select(geo.rings_overlap("many", hole))["many"].to_list() == expected
    assert df.select(geo.rings_overlap(hole, "many")).to_series().to_list() == expected


@pytest.mark.parametrize("single_first", [True, False])
def test_against_a_single_null_ring_everything_is_null(single_first: bool) -> None:
    df = pl.DataFrame(_rings("many", [_square(0, 0), _square(5, 5)]))
    one = _rings("one", [None])
    a, b = (one, "many") if single_first else ("many", one)

    assert df.select(geo.rings_overlap(a, b)).to_series().to_list() == [None, None]


def test_rings_in_different_crss_are_refused_at_plan_time() -> None:
    df = pl.DataFrame([_rings("a", [_square(0, 0)]), _rings("b", [_square(0, 0)])])
    lf = df.lazy().select(
        geo.rings_overlap(geo.set_crs("a", "EPSG:3857"), geo.set_crs("b", "EPSG:28992"))
    )

    with pytest.raises(ValueError, match="different CRSs"):
        lf.collect_schema()


def test_a_crs_on_only_one_ring_is_refused_at_plan_time() -> None:
    df = pl.DataFrame([_rings("a", [_square(0, 0)]), _rings("b", [_square(0, 0)])])
    lf = df.lazy().select(geo.rings_overlap(geo.set_crs("a", "EPSG:3857"), "b"))

    with pytest.raises(ValueError, match="only one of the rings"):
        lf.collect_schema()


def test_the_same_crs_on_both_is_fine() -> None:
    df = pl.DataFrame([_rings("a", [_square(0, 0)]), _rings("b", [_square(0, 0)])])
    out = df.select(
        geo.rings_overlap(geo.set_crs("a", "EPSG:3857"), geo.set_crs("b", "EPSG:3857"))
    )

    assert out["a"].to_list() == [True]


def test_agrees_with_shapely() -> None:
    """Small integer grids give many shared edges and corners."""
    rng = random.Random(0)

    def shape() -> list[tuple[float, float]]:
        x, y = rng.randint(0, 4), rng.randint(0, 4)
        if rng.random() < 0.5:
            w, h = rng.randint(1, 3), rng.randint(1, 3)
            return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        dx, dy = rng.randint(1, 3), rng.randint(1, 3)
        return [(x, y), (x + dx, y), (x, y + dy)]

    pairs = [(shape(), shape()) for _ in range(500)]
    a = [_ring(*first) for first, _ in pairs]
    b = [_ring(*second) for _, second in pairs]
    shapes = [
        (shapely.Polygon(first), shapely.Polygon(second)) for first, second in pairs
    ]

    assert _overlap(a, b) == [s.relate_pattern(t, "T********") for s, t in shapes]
    assert _overlap(a, b, include_touching=True) == [s.intersects(t) for s, t in shapes]


def test_the_namespace_forwards() -> None:
    df = pl.DataFrame([_rings("a", [_square(0, 0)]), _rings("b", [_square(1, 0)])])

    out = df.select(
        gpl.col("a").geo.rings_overlap("b"),
        gpl.col("a").geo.rings_overlap("b", include_touching=True).alias("touching"),
    )

    assert out.row(0) == (False, True)


@pytest.mark.parametrize("geometry", ["point", "polygon"])
def test_refuses_anything_but_linestrings(
    geometry: str, coords: pl.DataFrame, ring_coords: pl.DataFrame
) -> None:
    df = coords.select(XY.point()) if geometry == "point" else XY.polygons(ring_coords)

    with pytest.raises(TypeError, match="rings_overlap expects two linestring"):
        df.select(geo.rings_overlap(geometry, geometry))
