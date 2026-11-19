# Getting started

We present two equivalent APIs.
All functionality is available through both.

```python
# functional
from geopolars import geo

geo.translate("route", (1.0, 2.0))

# expressions using a namespace
import geopolars as gpl

gpl.col("route").geo.translate((1.0, 2.0))
```

Both are fully type-checked, but with an asterisk.
Plain `pl.col("route").geo...` works at runtime,
but a checker cannot see namespaces that `register_expr_namespace`
patches onto `pl.Expr`.
To avoid that, prefer using `gpl.col` to keep static typing.

For a list of all the geometry operations, look at the generated documentation!
They are written under the *functional* API, but hold exactly the same for the namespace API.
