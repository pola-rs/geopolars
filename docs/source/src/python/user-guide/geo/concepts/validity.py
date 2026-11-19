# --8<-- [start:df]
import polars as pl

from geopolars import PolygonXY, geo

df = pl.DataFrame(
    {
        "name": ["square", "bowtie", "sliver", "empty"],
        "parcel": [
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))",
            "POLYGON ((0 0, 1 0, 1 0, 0 0))",
            "POLYGON EMPTY",
        ],
    }
).with_columns(parcel=geo.from_wkt("parcel", PolygonXY))
# --8<-- [end:df]

# --8<-- [start:checks]
result = df.select(
    "name",
    geoarrow=geo.is_valid_geoarrow("parcel"),
    ogc=geo.is_valid_ogc("parcel"),
    empty=geo.is_empty("parcel"),
    repeats=geo.has_repeats_in_succession("parcel"),
)
print(result)
# --8<-- [end:checks]

# --8<-- [start:null_out]
result = df.with_columns(geo.null_out_invalid_ogc("parcel"))
print(result.with_columns(geo.to_wkt("parcel")))
# --8<-- [end:null_out]
