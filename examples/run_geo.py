import tempfile
from pathlib import Path

import polars as pl
from geopolars.datatypes import GeoPoint

import geopolars as gpl
from geopolars import PointXY, PointXYM, PointXYZ, PointXYZM, geometry

df = pl.DataFrame(
    {
        "city": ["Amsterdam", "Delft", "Utrecht"],
        "lon": [4.9041, 4.3571, 5.1214],
        "lat": [52.3676, 52.0116, 52.0907],
        # Elevation in metres. The Netherlands being what it is, one is below
        # sea level.
        "elevation": [-2.0, 1.0, 5.0],
        # An arbitrary per-vertex measure: metres travelled along a route.
        "distance": [0.0, 58_000.0, 94_000.0],
    }
)

print(f"We start with individual columns per axis: {df}")

# Build a point column from its coordinate columns.
# This is fully python, as it just helps construct an expression.
# assembled by ordinary expressions, then labelled with the extension dtype.
# Which of z and m you pass is what decides the dimension.
points = df.select(
    "city",
    xy=geometry.point("lon", "lat"),
    xyz=geometry.point("lon", "lat", z="elevation"),
    xym=geometry.point("lon", "lat", m="distance"),
    xyzm=geometry.point("lon", "lat", z="elevation", m="distance"),
)
print(f"We turn those into points! {points}")
print("Points are represented internally as structs.")
for name, expected in [
    ("xy", PointXY()),
    ("xyz", PointXYZ()),
    ("xym", PointXYM()),
    ("xyzm", PointXYZM()),
]:
    dtype = points.schema[name]
    print(f"{name:>5}: {dtype!r:<10} storage={dtype.ext_storage()}")
    assert dtype == expected, name
print()

print("XYM is not XYZ: a measure is not an elevation.")
print("  PointXYM() == PointXYZ() ->", PointXYM() == PointXYZ(), "\n")

print("You deliberately can't access the underlying data directly.")
try:
    points.select(pl.col("xyz").struct.field("z"))
except Exception as e:
    print(f"struct.field through the type -> {type(e).__name__}: {e}")
print("You can still do it through `col().ext.storage()`: ")
print(
    "You can still do it through `col().ext.storage()`: ",
    points.select(pl.col("xyz").ext.storage()).schema["xyz"],
    "\n",
)

print("Round trip through the plugin: the dtype has to survive, in every dimension.")
moved = points.select(
    "city",
    *[
        gpl.col(name).geometry.translate(dx=1.0, dy=-1.0, dz=10.0).alias(name)
        # dz only applies where there is a z to shift.
        if name in ("xyz", "xyzm")
        else gpl.col(name).geometry.translate(dx=1.0, dy=-1.0).alias(name)
        for name in ("xy", "xyz", "xym", "xyzm")
    ],
)
print(moved)
print("dtypes preserved:", moved.schema == points.schema)
print(
    "m carried through untouched:",
    moved["xym"].ext.storage().struct.field("m").to_list()
    == points["xym"].ext.storage().struct.field("m").to_list(),
    "\n",
)

print("A dz that could not do anything is an error, not a silent no-op.")
for name in ("xy", "xym"):
    try:
        points.select(gpl.col(name).geometry.translate(dx=0.0, dy=0.0, dz=1.0))
    except Exception as e:
        detail = next(l for l in str(e).splitlines() if "no z coordinate" in l)
        print(f"  {name:>5} -> {type(e).__name__}: {detail.strip()}")

print("\nYou can also do more generic Polars operations on these special types")
grouped = moved.group_by(pl.col("city").str.len_chars().alias("n")).agg("xyzm")
print("after group_by:", grouped.schema["xyzm"])
print("after sort:    ", moved.sort("xyzm").schema["xyzm"])

print("\nAlso through a parquet roundtrip")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "points.parquet"
    moved.write_parquet(path)
    back = pl.read_parquet(path)
for name in ("xy", "xyz", "xym", "xyzm"):
    print(f"  after parquet {name:>5}: {back.schema[name]!r}")
print("identical:", back.schema == moved.schema)

print(
    "\nWrong data should be reported while making the expression already, \
        before collecting the data"
)
try:
    df.lazy().select(gpl.col("lon").geometry.translate(dx=1.0, dy=1.0)).collect_schema()
except Exception as e:
    detail = next(l for l in str(e).splitlines() if "geoarrow.point" in l)
    print(f"non-point input -> {type(e).__name__}: {detail.strip()}")

print(
    "\nYou can only interpret Storage that follows the spec to an arrow extension type."
)
for storage in (
    # Order is significant: the spec fixes x before y...
    pl.Struct({"y": pl.Float64, "x": pl.Float64}),
    # ...and z before m.
    pl.Struct({"x": pl.Float64, "y": pl.Float64, "m": pl.Float64, "z": pl.Float64}),
    # The names are the dimension, so these are not coordinates.
    pl.Struct({"lon": pl.Float64, "lat": pl.Float64}),
    # Coordinates are doubles, and we will not quietly widen.
    pl.Struct({"x": pl.Float32, "y": pl.Float32}),
):
    try:
        GeoPoint.ext_from_params("geoarrow.point", storage, None)
    except ValueError as e:
        print(f"rejected -> {e}")

print("\nMetadata is carried over verbatim")
xyzm = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "m": pl.Float64})
carried = GeoPoint.ext_from_params("geoarrow.point", xyzm, '{"crs":"EPSG:4326"}')
print("dispatched to:", repr(carried))
print("metadata carried through:", carried.ext_metadata())
