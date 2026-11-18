===
Geo
===

The functions in `geopolars.geo` work on GeoArrow geometry columns.
Each of them is also available under the `geo` attribute of an expression,
such as `pl.col("shape").geo.area()`.

.. toctree::
   :maxdepth: 2

   construction
   encoding
   components
   measurement
   predicates
   validity
   transformation
   crs
   modification
   operations
