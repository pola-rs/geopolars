# --8<-- [start:df]
import polars as pl

import geopolars as gpl
from geopolars import LineStringXY, geo

df = pl.DataFrame(
    {"route": ["LINESTRING (0 0, 1 1, 2 0)", "LINESTRING (5 5, 6 5)"]}
).select(route=geo.from_wkt("route", LineStringXY))
# --8<-- [end:df]

# --8<-- [start:functional]
result = df.select(geo.translate("route", (1.0, 2.0)))
print(result.select(geo.to_wkt("route")))
# --8<-- [end:functional]

# --8<-- [start:namespace]
result = df.select(gpl.col("route").geo.translate((1.0, 2.0)))
print(result.select(geo.to_wkt("route")))
# --8<-- [end:namespace]
