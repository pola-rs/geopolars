"""Where the cost of the centroid abstraction actually goes.

`run_centroid_manual.py` shows the plan you would write by hand;
`run_centroid.py` shows the plan the abstractions build.
This one puts a number on each layer between the two, by adding them
one at a time to the same hand-written expression:

    minimal  ->  + validity checks  ->  + coalesce wrappers  ->  + ext.to/ext.storage

so it is visible which layer is the one worth removing.

Mind that those layers are stand-ins, not the real thing: the validity one is
much dearer than `_defined` actually is, because per-geometry overhead only
dominates at two vertices apiece. `breakdown()` at the bottom times the real
`_defined` and `_mean` instead, and puts the cost where it belongs.
"""

import functools
import operator
import time

import polars as pl

import geopolars as gpl
from geopolars import geometry
from geopolars.datatypes import GeoLineString
from geopolars.geometry import centroid as impl

AXES = ("x", "y", "m")
SAMPLES = pl.col("samples")


def frames(rows: int, vertices: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`rows` linestrings of `vertices` coordinates each, plain and as a geometry."""
    total = rows * vertices
    coords = pl.select(
        line=pl.int_range(total) // vertices,
        x=pl.int_range(total).cast(pl.Float64) * 0.37 % 100,
        y=pl.int_range(total).cast(pl.Float64) * 0.73 % 100,
        m=pl.int_range(total).cast(pl.Float64) * 1.11 % 100,
    )
    plain = (
        coords.lazy()
        .group_by("line", maintain_order=True)
        .agg(pl.struct(*AXES).alias("samples"))
        .select("samples")
        .collect()
    )
    geo = plain.lazy().select(SAMPLES.ext.to(gpl.LineStringXYM())).collect()
    return plain, geo


def fastest(frame: pl.LazyFrame, runs: int = 9) -> float:
    """Milliseconds for the quickest of `runs` collects, after a warm-up."""
    frame.collect()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        frame.collect()
        times.append(time.perf_counter() - start)
    return min(times) * 1000


def axis(column: pl.Expr, name: str) -> pl.Expr:
    """One axis of a linestring's coordinates, still one list per row."""
    return column.list.eval(pl.element().struct.field(name))


def means(column: pl.Expr) -> list[pl.Expr]:
    """The mean of each axis, per row. The whole of what a centroid is."""
    return [(axis(column, a).list.sum() / column.list.len()).alias(a) for a in AXES]


def layers(plain: pl.DataFrame, geo: pl.DataFrame) -> dict[str, pl.LazyFrame]:
    """The same centroid, with one more layer of abstraction each time."""
    nothing = pl.lit(None, dtype=pl.Float64)

    # 1. What the centroid is, and nothing else.
    minimal = plain.lazy().select(centre=pl.struct(means(SAMPLES)))

    # 2. ...plus `_defined`: one null scan per axis, on top of the sums.
    whole = SAMPLES.list.len() > 0
    for a in AXES:
        whole = whole & (axis(SAMPLES, a).list.count_matches(None) == 0)
    checked = plain.lazy().select(
        centre=pl.when(SAMPLES.is_not_null() & whole).then(pl.struct(means(SAMPLES)))
    )

    # 3. ...plus what `by_geometry` wraps every branch in. By the time the
    # optimiser has run, the selectors have pruned all but one branch, so every
    # one of these coalesces has a single argument left and is a no-op.
    wrapped = plain.lazy().select(
        centre=pl.coalesce(
            [
                pl.when(pl.coalesce([SAMPLES.is_not_null() & whole, pl.lit(False)])).then(
                    pl.struct([pl.coalesce([m, nothing]) for m in means(SAMPLES)])
                )
            ]
        )
    )

    # 4. ...plus the extension type itself, taken off and put back on.
    storage = SAMPLES.ext.storage()
    intact = storage.list.len() > 0
    for a in AXES:
        intact = intact & (axis(storage, a).list.count_matches(None) == 0)
    typed = geo.lazy().select(
        centre=pl.coalesce(
            [
                pl.when(pl.coalesce([SAMPLES.is_not_null() & intact, pl.lit(False)]))
                .then(pl.struct([pl.coalesce([m, nothing]) for m in means(storage)]))
                .ext.to(gpl.PointXYM())
            ]
        )
    )

    return {
        "1. minimal (what a centroid is)": minimal,
        "2.  + the validity checks": checked,
        "3.  + the dispatch coalesces": wrapped,
        "4.  + ext.storage() / ext.to()": typed,
        "5. geometry.coordinate_centroid": geo.lazy().select(
            centre=geometry.coordinate_centroid("samples")
        ),
    }


def report(rows: int, vertices: int) -> None:
    plain, geo = frames(rows, vertices)
    print(f"\n{rows:,} linestrings of {vertices} coordinates ({rows * vertices:,} total)")
    print("-" * 62)
    base = None
    for name, frame in layers(plain, geo).items():
        ms = fastest(frame)
        base = base if base is not None else ms
        print(f"  {name:<40}{ms:7.1f} ms   {ms / base:4.2f}x")


def breakdown(rows: int, vertices: int) -> None:
    """The real expression, piece by piece, against the two bounds around it."""
    _, geo = frames(rows, vertices)
    column, storage = pl.col("samples"), pl.col("samples").ext.storage()
    dtype = GeoLineString.of_dimension(AXES)

    pieces = {
        "_coordinates (the emptiness check)": impl._coordinates(dtype, column) > 0,
        # What `_defined` used to do on every call, now a construction-time
        # guarantee: one walk over every coordinate, once per axis.
        "the null scan that used to sit here": functools.reduce(
            operator.and_,
            (impl._axis(storage, 1, a).list.count_matches(None) == 0 for a in AXES),
        ),
        "_mean, every axis": pl.struct(
            [impl._mean(dtype, column, a).alias(a) for a in AXES]
        ),
        "...of which is axis extraction": pl.struct(
            [storage.list.eval(pl.element().struct.field(a)).alias(a) for a in AXES]
        ),
        "coordinate_centroid (all of it)": geometry.coordinate_centroid("samples"),
    }

    print(f"\n{rows:,} linestrings of {vertices} coordinates, {''.join(AXES)}")
    print("-" * 62)
    for name, piece in pieces.items():
        print(f"  {name:<40}{fastest(geo.lazy().select(out=piece)):7.1f} ms")


# Short geometries are the worst case: the per-row wrappers are paid over and
# over, while the sums they wrap have almost nothing to add up.
report(rows=2_000_000, vertices=2)
# Long ones drown the wrappers in real work.
report(rows=4_000, vertices=1_000)

# Where the time really goes: `list.sum()` inside `_mean`, not the null scan.
breakdown(rows=200_000, vertices=20)

# The node the whole thing streams through, or does not.
# `run_centroid.py` falls back to an in-memory node -- but that node is the
# `implode()` that gathers the rows into one geometry, not the centroid.
# Plain Polars does exactly the same thing, with no geometry in sight:
plain, geo = frames(rows=1_000, vertices=4)
print("\nThe centroid alone, given a geometry column that already exists:")
centre = geo.lazy().select(centre=geometry.coordinate_centroid("samples"))
print(centre.explain(engine="streaming"))
centre.show_graph(plan_stage="physical", engine="streaming", optimized=True)

print("\nAnd `implode()` on a plain float column, for comparison:")
pl.LazyFrame({"a": [1.0, 2.0]}).select(pl.col("a").implode()).show_graph(
    plan_stage="physical", engine="streaming", optimized=True
)
