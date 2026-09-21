"""Manual implementation of getting the centroid of some coordinates.
This is here so you can compare the plan of something hand-built to using
the geopolars abstractions layer.
"""

import polars as pl

# The same four soil samples as in `run_centroid.py`,
# as three plain float columns.
lf = pl.LazyFrame(
    {
        "x": [0.0, 4.0, 4.0, 0.0],
        "y": [0.0, 0.0, 3.0, 3.0],
        "value": [10.0, 20.0, 30.0, 40.0],
    }
)

print("One row per sample:")
print(lf.collect())

flat = lf.select(
    pl.col("x").mean(),
    pl.col("y").mean(),
    pl.col("value").mean(),
)

print("\nCentroid of the four samples, straight off the loose columns:")
print(flat.collect())
print(flat.explain(engine="streaming"))
flat.show_graph(plan_stage="physical", engine="streaming", optimized=True)

# The coordinate struct a point wraps, built in-line from the loose columns.
# `m` is a measure rather than an axis, but it averages like one,
# so all three go in together.
point = pl.struct(x=pl.col("x"), y=pl.col("y"), m=pl.col("value"))

# One mean per axis, mapped over the fields of that struct.
centroid = lf.select(
    centre=pl.struct(
        # [point.struct.field(axis).mean().alias(axis) for axis in ("x", "y", "m")]
        point.struct.field("x").mean().alias("x_mean"),
        point.struct.field("y").mean().alias("y_mean"),
        point.struct.field("m").mean().alias("m_mean"),
    )
)
# and now we need to convert it back to its point type again.

print(
    "\nNow we put them all in a struct to get a little closer to the abstraction \
    that GeoPolars uses:"
)
print(centroid.collect())
print(centroid.explain(engine="streaming"))
centroid.show_graph(plan_stage="physical", engine="streaming", optimized=True)
