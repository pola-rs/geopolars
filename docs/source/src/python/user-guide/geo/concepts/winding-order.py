# --8<-- [start:df]
import polars as pl

from geopolars import PolygonXY, geo

df = pl.DataFrame(
    {
        "name": ["counter-clockwise", "clockwise"],
        "parcel": [
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0), (0.5 0.5, 0.5 1, 1 1, 1 0.5, 0.5 0.5))",
            "POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))",
        ],
    }
).with_columns(parcel=geo.from_wkt("parcel", PolygonXY))
# --8<-- [end:df]

# --8<-- [start:is_ccw]
print(df.select("name", ccw=geo.is_ccw("parcel"), area=geo.area("parcel")))
# --8<-- [end:is_ccw]

# --8<-- [start:to_ccw]
result = df.select("name", parcel=geo.to_ccw("parcel"))
print(result.with_columns(ccw=geo.is_ccw("parcel"), parcel=geo.to_wkt("parcel")))
# --8<-- [end:to_ccw]

# --8<-- [start:new_rings]
result = df.select(
    "name",
    kept=geo.buffer("parcel", 0.5),
    ccw=geo.buffer("parcel", 0.5, new_ring_orientation="ccw"),
)
print(result.select("name", geo.is_ccw("kept"), geo.is_ccw("ccw")))
# --8<-- [end:new_rings]
