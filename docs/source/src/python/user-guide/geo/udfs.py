# --8<-- [start:df]
import polars as pl

from geopolars import PolygonXY, geo

df = pl.DataFrame({"parcel": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"]}).with_columns(
    parcel=geo.from_wkt("parcel", PolygonXY, crs="EPSG:28992")
)
# --8<-- [end:df]

# --8<-- [start:map_elements]
result = df.with_columns(
    pl.col("parcel").map_elements(
        lambda v: [[{**p, "x": p["x"] * 2} for p in ring] for ring in v.to_list()],
        return_dtype=df.schema["parcel"],
    )
)
print(result.with_columns(geo.to_wkt("parcel")))
# --8<-- [end:map_elements]
