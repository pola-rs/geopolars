# --8<-- [start:declare]
import polars as pl

from geopolars import LineStringXY, geo

cities = pl.DataFrame(
    {
        "city": ["Amsterdam", "Utrecht"],
        "lon": [4.9041, 5.1214],
        "lat": [52.3676, 52.0907],
    }
).select("city", geometry=geo.point("lon", "lat", crs="EPSG:4326"))
print(cities)
# --8<-- [end:declare]

# --8<-- [start:set_crs]
unlabelled = pl.DataFrame({"lon": [4.9041], "lat": [52.3676]}).select(
    geometry=geo.point("lon", "lat")
)
labelled = unlabelled.with_columns(geo.set_crs("geometry", "EPSG:4326"))
print(unlabelled.schema)
print(labelled.schema)
# --8<-- [end:set_crs]

# --8<-- [start:to_crs]
# EPSG:28992 is the Dutch national grid, in metres.
dutch = cities.with_columns(geo.to_crs("geometry", "EPSG:28992"))
print(dutch)
# --8<-- [end:to_crs]

# --8<-- [start:measure]
amsterdam = geo.point(pl.lit(4.9041), pl.lit(52.3676), crs="EPSG:4326")
result = cities.select(
    "city",
    geodesic=geo.distance(amsterdam, "geometry"),
    planar=geo.distance(amsterdam, "geometry", method="planar"),
)
print(result)
# --8<-- [end:measure]

# --8<-- [start:mismatch]
from polars.exceptions import ComputeError

both = cities.with_columns(dutch=dutch["geometry"])
try:
    both.select(geo.distance("geometry", "dutch"))
except ComputeError as err:
    print(err)
# --8<-- [end:mismatch]

# --8<-- [start:antimeridian]
routes = pl.DataFrame(
    {
        "name": ["Fiji", "Bering Strait"],
        "route": ["LINESTRING (178 -17, 182 -17)", "LINESTRING (190 65, 195 66)"],
    }
).with_columns(route=geo.from_wkt("route", LineStringXY, crs="EPSG:4326"))

result = routes.select(
    "name",
    wrapped=geo.wrap_longitude("route"),
    skip_crossing=geo.wrap_longitude("route", skip_crossing=True),
).with_columns(geo.to_wkt("wrapped"), geo.to_wkt("skip_crossing"))
print(result)
# --8<-- [end:antimeridian]

# --8<-- [start:bounds]
print(routes.select("name", bounds=geo.bounds("route")))
# --8<-- [end:bounds]
