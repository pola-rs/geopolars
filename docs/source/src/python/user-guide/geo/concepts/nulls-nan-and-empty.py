# --8<-- [start:df]
import polars as pl

from geopolars import PolygonXY, geo

df = pl.DataFrame(
    {
        "name": ["square", "empty", "null"],
        "parcel": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POLYGON EMPTY", None],
    }
).with_columns(parcel=geo.from_wkt("parcel", PolygonXY))

result = df.select(
    "name",
    is_null=pl.col("parcel").is_null(),
    is_empty=geo.is_empty("parcel"),
    area=geo.area("parcel"),
    centroid=geo.to_wkt(geo.centroid("parcel")),
)
print(result)
# --8<-- [end:df]

# --8<-- [start:null_inside]
points = pl.DataFrame({"x": [1.0, None], "y": [2.0, 3.0]}).with_columns(
    point=geo.point("x", "y")
)
print(points)
# --8<-- [end:null_inside]

# --8<-- [start:empty_point]
nan = float("nan")
points = pl.DataFrame({"x": [1.0, nan, nan], "y": [2.0, nan, 3.0]}).with_columns(
    point=geo.point("x", "y")
)
print(
    points.with_columns(
        empty=geo.is_empty("point"),
        finite=geo.is_valid_geoarrow("point", check_finite_coordinates=True),
        wkt=geo.to_wkt("point"),
    )
)
# --8<-- [end:empty_point]

# --8<-- [start:nested]
routes = pl.DataFrame(
    {
        "name": ["fine", "gap", "only gaps"],
        "x": [[0.0, 1.0, 2.0], [0.0, nan, 2.0], [nan, nan]],
        "y": [[0.0, 1.0, 0.0], [0.0, nan, 0.0], [nan, nan]],
    }
).select("name", route=geo.line_string_from_columns("x", "y"))
result = routes.select(
    "name",
    is_empty=geo.is_empty("route"),
    nested=geo.is_empty("route", check_nested=True),
    length=geo.length("route"),
)
print(result)
# --8<-- [end:nested]

# --8<-- [start:geos]
from polars.exceptions import ComputeError

try:
    routes.select(geo.simplify("route", 0.1))
except ComputeError as err:
    print(err)

result = routes.select(
    "name", simplified=geo.simplify("route", 0.1, on_row_error="null")
)
print(result.with_columns(geo.to_wkt("simplified")))
# --8<-- [end:geos]

# --8<-- [start:filter]
result = routes.select("name", route=geo.filter_coordinates("route"))
print(result.with_columns(geo.to_wkt("route")))
# --8<-- [end:filter]
