# User-defined functions

Because GeoPolars has GeoArrow data types as first-class citizens,
you can use Polars' `map_elements` for a user-defined function (UDF)
like you would for other Polars data types.
There are some small extra things to consider though!

Inside a Python UDF, what the function receives depends on the geometry type.

| Type           | Value passed to the function         |
| -------------- | ------------------------------------ |
| PointXY        | `{'x': 1.0, 'y': 2.0}` (a dict)      |
| LineStringXY   | a Polars Series of `struct{x, y}`      |
| PolygonXY      | a Series of `list[struct{x, y}]`       |
| MultiPolygonXY | a Series of `list[list[struct{x, y}]]` |

Points arrive as plain Python, but everything else arrives as a Polars Series.
If you want to work with rings and coordinates, you'll have to call `.to_list()`.
After that, editing works, and returning nested lists of dicts rebuilds the geometry:

```python
pl.col("g").map_elements(
    lambda v: [[{**p, "x": p["x"] * 2} for p in ring] for ring in v.to_list()],
    return_dtype=df.schema["g"],
)
```

Two things in this snippet are easy to get wrong:

- `return_dtype` must be an instance. Passing the class like `return_dtype=PolygonXY`, raises a TypeError.
- The CRS can be lost. `PolygonXY()` produces a polygon column with no CRS. An existing CRS can be retained by using `df.schema["g"]`.

TODO: add helper function to set the CRS of the original column and describe it here.
