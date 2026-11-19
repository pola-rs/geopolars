# --8<-- [start:write]
import polars as pl

import geopolars as gpl
from geopolars import geo

cities = pl.DataFrame(
    {
        "city": ["Amsterdam", "Delft", "Utrecht"],
        "lon": [4.9041, 4.3571, 5.1214],
        "lat": [52.3676, 52.0116, 52.0907],
    }
).select("city", geometry=geo.point("lon", "lat", crs="EPSG:4326"))

cities.write_parquet(
    "docs/assets/data/cities.parquet", metadata=gpl.geoparquet_metadata(cities)
)
# --8<-- [end:write]

# --8<-- [start:read]
path = "docs/assets/data/cities.parquet"
cities = pl.scan_parquet(path).pipe(gpl.from_geoparquet, path)
print(cities.collect())
# --8<-- [end:read]

# --8<-- [start:wkb]
from geopolars import PointXY

path = "docs/assets/data/cities_wkb.parquet"
encoded = cities.with_columns(geo.to_wkb("geometry"))
encoded.sink_parquet(path, metadata=gpl.geoparquet_metadata(encoded))

wkb = pl.scan_parquet(path).pipe(gpl.from_geoparquet, path)
print(wkb.collect())
print(wkb.select("city", geo.from_wkb("geometry", PointXY)).collect())
# --8<-- [end:wkb]

# --8<-- [start:covering]
covered = cities.pipe(gpl.with_covering)
covered.sink_parquet(
    "docs/assets/data/cities_bbox.parquet",
    metadata=gpl.geoparquet_metadata(covered, covering=True),
)
print(covered.collect())
# --8<-- [end:covering]

# --8<-- [start:crs]
amsterdam = geo.point(pl.lit(4.9041), pl.lit(52.3676), crs="EPSG:4326")
relabelled = cities.with_columns(geo.set_crs("geometry", "EPSG:4326", force=True))
result = relabelled.select("city", km=geo.distance("geometry", amsterdam) / 1000)
print(result.collect())
# --8<-- [end:crs]

# --8<-- [start:refuse]
from polars.exceptions import SchemaError

from geopolars import LineStringXYM

tracks = pl.DataFrame({"track": ["LINESTRING M (0 0 1, 1 1 2)"]}).select(
    track=geo.from_wkt("track", LineStringXYM)
)
try:
    gpl.geoparquet_metadata(tracks)
except SchemaError as err:
    print(err)
# --8<-- [end:refuse]

# --8<-- [start:wkt]
from io import StringIO

from geopolars import MultiPolygonXY

csv = StringIO(
    """name,geometry
square,"POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
two squares,"MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))"
"""
)
parcels = pl.read_csv(csv).with_columns(
    geometry=geo.from_wkt("geometry", MultiPolygonXY, crs="EPSG:28992")
)
print(parcels)
# --8<-- [end:wkt]
