# Getting started

GeoPolars stores geometries in GeoArrow columns.
Below, we read a column of WKT into linestrings.

{{code_block('user-guide/geo/getting-started','df',['DataFrame','geo.from_wkt','LineStringXY'])}}

```python exec="on" result="text" session="user-guide/geo/getting-started"
--8<-- "python/user-guide/geo/getting-started.py:df"
```

We present two equivalent APIs, and all functionality is available through both.
The first is a *functional* API, in the `geo` module:

{{code_block('user-guide/geo/getting-started','functional',['geo.translate','geo.to_wkt'])}}

```python exec="on" result="text" session="user-guide/geo/getting-started"
--8<-- "python/user-guide/geo/getting-started.py:functional"
```

The second uses the `geo` namespace on expressions:

{{code_block('user-guide/geo/getting-started','namespace',['gpl.col','geo.translate'])}}

```python exec="on" result="text" session="user-guide/geo/getting-started"
--8<-- "python/user-guide/geo/getting-started.py:namespace"
```

!!! tip "Static typing"

    Both APIs are fully type-checked, but with an asterisk.
    Plain `pl.col("route").geo...` works at runtime,
    but a type checker cannot see namespaces that `register_expr_namespace` patches onto `pl.Expr`.
    Use `gpl.col` instead, to keep static typing.

The API reference lists every geometry operation.
They are written under the functional API, but hold exactly the same for the namespace API.
