# geopolars-compute

Kernels for GeoArrow.
They take the views of [`geopolars-views`](../geopolars-views/README.md).
They return views for geometries, and Polars arrays otherwise.

The crate holds:

- the metrics that kernels measure with (`Planar`, `GeodesicMetric`).
- kernels for simple operations implemented directly on the arrow type in `native`, behind the `native` feature.
- kernels forwarding to GEOS in `geos`, behind the `geos` feature.
- kernels forwarding to rsgeo (the Rust `geo` crate) in `rsgeo`, behind the `rsgeo` feature.
- kernels forwarding to PROJ in `proj`.
- `is_valid_geoarrow`'s kernels in `geoarrow_validity`, which check the coordinates and the rings.
  These need the `native` feature.
- `is_valid_ogc`'s kernels in `ogc_validity`, which run the checks of `native`.
  Then they run those of `geos`, on the rows that pass.
  These need both features.
- the pass that moves every null value below a row out to the row in `inner_nulls`,
  or marks the parts holding one in a mask instead.
- the fold of each level's mask up to the rows in `passing`.
  `inner_nulls` and the validity kernels use it.

See the [docs](../docs/Kernel implementation guide.md) for more details.
