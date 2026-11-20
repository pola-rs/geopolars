# User-defined functions

Because GeoPolars has GeoArrow data types as first-class citizens,
you can use Polars' `map_elements` for a user-defined function (UDF)
like you would for other Polars data types.
There are some small extra things to consider though!

Inside a Python UDF, what the function receives depends on the [geometry type](concepts/geometry-types.md).

| Type | Value passed to the function |
| - | - |
| PointXY | `{'x': 1.0, 'y': 2.0}` (a dict) |
| LineStringXY | a Polars Series of `struct{x, y}` |
| PolygonXY | a Series of `list[struct{x, y}]` |
| MultiPolygonXY | a Series of `list[list[struct{x, y}]]` |

Points arrive as plain Python, but everything else arrives as a Polars Series.
If you want to work with rings and coordinates, you'll have to call `.to_list()`.
After that, editing works, and returning nested lists of dicts rebuilds the geometry.
Below, we stretch a polygon to twice its width:

{{code_block('user-guide/geo/udfs','df',['DataFrame','geo.from_wkt','PolygonXY'])}}

{{code_block('user-guide/geo/udfs','map_elements',['map_elements','geo.to_wkt'])}}

```python exec="on" result="text" session="user-guide/geo/udfs"
--8<-- "python/user-guide/geo/udfs.py:df"
--8<-- "python/user-guide/geo/udfs.py:map_elements"
```

!!! warning

    Two things in this snippet are easy to get wrong:

    - `return_dtype` must be an instance.
      Passing the class, like `return_dtype=PolygonXY`, raises a TypeError.
    - The CRS can be lost.
      `PolygonXY()` produces a polygon column with no [CRS](concepts/crs.md).
      Use `df.schema["parcel"]` to keep the existing CRS,
      or declare it again afterwards with `geo.set_crs`.
