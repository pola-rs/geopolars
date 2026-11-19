# --8<-- [start:df]
import polars as pl

from geopolars import PolygonXY, geo

df = pl.DataFrame(
    {
        "name": ["square", "triangle"],
        "parcel": [
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((0 0, 2 0, 1 1, 0 0))",
        ],
    }
).with_columns(parcel=geo.from_wkt("parcel", PolygonXY, crs="EPSG:28992"))
print(df)
print(df.schema)
# --8<-- [end:df]

# --8<-- [start:promote]
from geopolars import MultiPolygonXY

mixed = pl.DataFrame(
    {
        "parcel": [
            "POLYGON ((0 0, 1 0, 1 1, 0 0))",
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))",
        ]
    }
).select(parcel=geo.from_wkt("parcel", MultiPolygonXY))
for wkt in mixed.select(geo.to_wkt("parcel"))["parcel"]:
    print(wkt)
# --8<-- [end:promote]

# --8<-- [start:mismatch]
from polars.exceptions import ComputeError

try:
    pl.DataFrame({"route": ["LINESTRING (0 0, 1 1)"]}).select(
        geo.from_wkt("route", PolygonXY)
    )
except ComputeError as err:
    print(err)
# --8<-- [end:mismatch]

# --8<-- [start:dimensions]
from geopolars import LineStringXYZ

hike = pl.DataFrame({"route": ["LINESTRING Z (0 0 10, 100 0 15, 100 100 40)"]}).select(
    route=geo.from_wkt("route", LineStringXYZ)
)
print(hike.with_columns(elevation=geo.z("route"), flat=geo.force_2d("route")))
# --8<-- [end:dimensions]

# --8<-- [start:box]
boxes = df.select("name", bounds=geo.bounds("parcel"))
print(boxes)
print(boxes.select("name", geo.to_wkt(geo.to_polygon("bounds"))))
# --8<-- [end:box]

# --8<-- [start:encoded]
encoded = df.select(wkt=geo.to_wkt("parcel"), wkb=geo.to_wkb("parcel"))
print(encoded)
print(encoded.schema)
# --8<-- [end:encoded]
