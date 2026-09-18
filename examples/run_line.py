import tempfile
from pathlib import Path

import polars as pl
from geopolars.datatypes import GeoLineString, GeoPoint

import geopolars as gpl
from geopolars import (
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    LineStringXYZM,
    PolygonXY,
    geometry,
)

df = pl.DataFrame(
    {
        # Two cycling routes, sampled at a handful of points each. The routes
        # are of different lengths, which is the whole reason a linestring needs
        # a list: the offsets say where one route ends and the next begins.
        "route": ["north", "north", "north", "coast", "coast"],
        "lon": [4.9041, 4.8000, 4.7000, 3.6000, 3.5000],
        "lat": [52.3676, 52.4000, 52.5000, 51.5000, 51.4000],
        # Elevation in metres, and metres travelled along the route.
        "elevation": [-2.0, 1.0, 5.0, 0.0, 2.0],
        "distance": [0.0, 8_000.0, 19_000.0, 0.0, 12_000.0],
    }
)

print(f"We start with one row per vertex: {df}")

# A linestring is built from vertices that are already grouped: one list of
# points per line. `group_by().agg()` is what does the grouping, and
# `.implode()` inside it is what hands a whole group over as the one list a
# linestring is made of.
lines = df.group_by("route", maintain_order=True).agg(
    xy=geometry.linestring_from_vertices(geometry.point("lon", "lat").implode()),
    xyz=geometry.linestring_from_vertices(
        geometry.point("lon", "lat", z="elevation").implode()
    ),
    xym=geometry.linestring_from_vertices(
        geometry.point("lon", "lat", m="distance").implode()
    ),
    xyzm=geometry.linestring_from_vertices(
        geometry.point("lon", "lat", z="elevation", m="distance").implode()
    ),
)
print(f"Grouping the vertices gives linestrings: {lines}")
print("A linestring stores `List<Coordinate>`: the same coordinates, nested once.")
for name, expected in [
    ("xy", LineStringXY()),
    ("xyz", LineStringXYZ()),
    ("xym", LineStringXYM()),
    ("xyzm", LineStringXYZM()),
]:
    dtype = lines.schema[name]
    print(f"{name:>5}: {dtype!r:<15} storage={dtype.ext_storage()}")
    assert dtype == expected, name
print()

print("The vertices decide the dimension; nothing has to be spelled out twice.")
print("  a list of PointXYZ ->", repr(lines.schema["xyz"]), "\n")

print("Vertices are one way in. Coordinates that are already grouped are another:")
print("pass a column per axis, exactly as to `point`, and skip the vertices.")
print("`geometry.linestring` picks between the two by how you call it.")
grouped = df.group_by("route", maintain_order=True).agg("lon", "lat", "elevation")
print(grouped)
direct = grouped.select(
    "route",
    xy=geometry.linestring_from_columns("lon", "lat"),
    xyz=geometry.linestring_from_columns("lon", "lat", z="elevation"),
)
print(direct)
print("the same lines:", direct["xy"].equals(lines["xy"]), "\n")

print("Which of z and m you pass decides the dimension, as it does for a point.")
print("  ", repr(direct.schema["xy"]), "vs", repr(direct.schema["xyz"]), "\n")

print("The coordinate columns have to nest the same way, or there are no")
print("vertices to be had.")
try:
    grouped.select(geometry.linestring_from_columns("lon", pl.col("lat").list.head(1)))
except Exception as e:
    detail = next(l for l in str(e).splitlines() if "nest the same way" in l)
    print(f"  {type(e).__name__}: {detail.strip()}")
print()

print("A linestring is not a point, whatever its dimension.")
print("  LineStringXY() == PointXY() ->", LineStringXY() == gpl.PointXY(), "\n")

print("The offsets are what the extra nesting buys: 5 vertices, 2 routes.")
print(lines.select("route", n=pl.col("xy").ext.storage().list.len()), "\n")

print("Translating moves every vertex and leaves the routes as they were.")
moved = lines.select(
    "route",
    *[
        gpl.col(name).geometry.translate(dx=1.0, dy=-1.0, dz=10.0).alias(name)
        # dz only applies where there is a z to shift.
        if name in ("xyz", "xyzm")
        else gpl.col(name).geometry.translate(dx=1.0, dy=-1.0).alias(name)
        for name in ("xy", "xyz", "xym", "xyzm")
    ],
)
print(moved)
print("dtypes preserved:", moved.schema == lines.schema)
print(
    "m carried through untouched:",
    moved["xym"].ext.storage().explode(empty_as_null=False).struct.field("m").to_list()
    == lines["xym"]
    .ext.storage()
    .explode(empty_as_null=False)
    .struct.field("m")
    .to_list(),
    "\n",
)

print("GeoArrow allows nulls only at the outermost level.")
vertex_list = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
edge_cases = pl.DataFrame(
    {
        "vertices": [
            [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
            # An empty line is a geometry a linestring can hold, unlike a point.
            [],
            None,
            # A missing vertex, or a vertex missing a coordinate, takes the
            # whole line with it.
            [{"x": 0.0, "y": 0.0}, None],
            [{"x": 0.0, "y": None}],
        ]
    },
    schema={"vertices": vertex_list},
).select(line=geometry.linestring_from_vertices("vertices"))
print(edge_cases)
print("null:", edge_cases["line"].is_null().to_list(), "\n")

print("Vertices need not be points yet: bare coordinate structs read the same.")
print("  ", edge_cases.schema["line"], "\n")

print("Also through a parquet roundtrip")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "lines.parquet"
    moved.write_parquet(path)
    back = pl.read_parquet(path)
for name in ("xy", "xyz", "xym", "xyzm"):
    print(f"  after parquet {name:>5}: {back.schema[name]!r}")
print("identical:", back.schema == moved.schema, "\n")

print("Wrong input is reported while building the expression, before collecting.")
for column in ("route", "xy"):
    try:
        lines.lazy().select(geometry.linestring_from_vertices(column)).collect_schema()
    except Exception as e:
        detail = next(l for l in str(e).splitlines() if "expected" in l)
        print(f"  {column:>5} -> {type(e).__name__}: {detail.strip()}")

print("\nYou can only read storage that follows the spec as a linestring.")
for storage in (
    # A linestring nests its coordinates; a bare coordinate is a point.
    pl.Struct({"x": pl.Float64, "y": pl.Float64}),
    # Order is significant inside the coordinate, one level down.
    pl.List(pl.Struct({"y": pl.Float64, "x": pl.Float64})),
    # Interleaved coordinates are a layout this version does not implement.
    pl.List(pl.Float64),
):
    try:
        GeoLineString.ext_from_params("geoarrow.linestring", storage, None)
    except ValueError as e:
        print(f"rejected -> {e}")

print("\nMetadata comes along when points are converted into linestrings,")
print("which is what the spec asks for `edges`.")
spherical = GeoPoint.ext_from_params(
    "geoarrow.point", gpl.PointXY().ext_storage(), '{"edges":"spherical"}'
)
carried = (
    df.select(vertex=pl.struct(x=pl.col("lon"), y=pl.col("lat")).ext.to(spherical))
    .select(pl.col("vertex").implode())
    .select(line=geometry.linestring_from_vertices("vertex"))
)
print("vertices:", repr(spherical), spherical.ext_metadata())
print("line:    ", repr(carried.schema["line"]), carried.schema["line"].ext_metadata())

print("\nA ring is a closed linestring, and a polygon is a list of those.")
rings = pl.DataFrame(
    {
        # One square plot with a triangular hole in it, and one plain triangle.
        # The first ring of a polygon is its exterior; the rest are holes.
        "plot": ["field"] * 9 + ["yard"] * 4,
        "ring": ["outer"] * 5 + ["hole"] * 4 + ["outer"] * 4,
        # Every ring repeats its first vertex last. `polygon` does not check
        # that, nor close a ring for you.
        "lon": [0.0, 4.0, 4.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 10.0, 12.0, 11.0, 10.0],
        "lat": [0.0, 0.0, 3.0, 3.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0],
    }
)
print(f"One row per vertex, now with a ring to belong to: {rings}")

# Two groupings: vertices into rings, then rings into polygons.
boundaries = rings.group_by("plot", "ring", maintain_order=True).agg(
    boundary=geometry.linestring_from_vertices(geometry.point("lon", "lat").implode())
)
print(f"The rings are ordinary linestrings: {boundaries}")

plots = boundaries.group_by("plot", maintain_order=True).agg(
    boundary=geometry.polygon_from_rings(pl.col("boundary").implode())
)
print(f"Which a second grouping turns into polygons: {plots}")
print("A polygon stores `List<List<Coordinate>>`: the same coordinates, twice nested.")
print(f"  {plots.schema['boundary']!r}: {plots.schema['boundary'].ext_storage()}")
assert plots.schema["boundary"] == PolygonXY()

print("\nThe nesting is the only difference: rings per polygon, vertices per ring.")
print(
    plots.select(
        "plot",
        rings=pl.col("boundary").ext.storage().list.len(),
        vertices=pl.col("boundary")
        .ext.storage()
        .list.eval(pl.element().list.len())
        .list.sum(),
    )
)

print("\nCoordinate columns work a level deeper too: one list per polygon, of")
print("one list per ring. The two groupings are the two levels of nesting.")
direct_plots = (
    rings.group_by("plot", "ring", maintain_order=True)
    .agg("lon", "lat")
    .group_by("plot", maintain_order=True)
    .agg("lon", "lat")
    .select("plot", boundary=geometry.polygon_from_columns("lon", "lat"))
)
print(direct_plots)
print("the same polygons:", direct_plots["boundary"].equals(plots["boundary"]))
