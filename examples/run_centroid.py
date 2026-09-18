"""Example showing the calculation of centroids of coordinates.
Shows the following steps:
- changing columns into points
- calculating centroids from the points.
Ideally this should just be an abstraction on the plugin layer.
It uses only polars under the hood, no custom nodes and no magic!
"""

import polars as pl

import geopolars as gpl
from geopolars import PointXYM, geometry

# Four soil samples: where they were taken, and what was measured there.
# `value` is the measure a point carries as `m`.
lf = pl.LazyFrame(
    {
        "x": [0.0, 4.0, 4.0, 0.0],
        "y": [0.0, 0.0, 3.0, 3.0],
        "value": [10.0, 20.0, 30.0, 40.0],
    }
)

print("Three plain float columns, one row per sample:")
print(lf.collect())

# Passing `m` is what makes these XYM points: x and y say where,
# `value` says  what was measured there.
points = lf.select(point=geometry.point("x", "y", m="value"))

print("\nOne point per row, with the measurement carried along:")
print(points.collect())
print("dtype:", points.collect_schema()["point"])
assert points.collect_schema()["point"] == PointXYM()

collection = points.select(pl.col("point").implode()).select(
    # TODO: replace with multipoint
    samples=geometry.linestring("point")
)

print("\nGathered into one geometry, the four samples are one row:")
print(collection.collect())

centre = collection.select(centre=geometry.coordinate_centroid("samples"))

print(
    "The plan behind it: Ideally, all abstractions fall away, \
        and it is as cheap as manually building everything \
        after Polars is allowed to optimise it."
)
print(centre.explain())
print("\nThe physical plan, node by node:")
centre.show_graph(plan_stage="physical", engine="streaming", optimized=True)

print("\nThe centroid of the four samples:")
print(centre.collect())

print("\nThe same numbers, averaged column by column:")
print(
    lf.select(pl.col("x").mean(), pl.col("y").mean(), pl.col("value").mean()).collect()
)

print("\nThe whole thing is one lazy query, so it streams:")
print(centre.collect(engine="streaming"))

print("\nThe namespace spelling is the same expression:")
print(
    collection.select(
        centre=gpl.col("samples").geometry.coordinate_centroid()
    ).collect()
)
