# GeoPolars

In this branch, we are rewriting large chunks of GeoPolars to work much more closely with the rest of Polars.
This is all very much still work-in-progress, and as such this branch is HIGHLY unstable.
Keep an eye on the tracking issues with the dev tag for where we are right now.
We'll need a little bit more time to make things nice, but we're making a lot of progress very quickly,
so stay tuned!

## Overview of Features

GeoPolars has Geo data types as first-class Polars citizens,
  represented using the Arrow Extension format as [GeoArrow](https://geoarrow.org/).
  This has several advantages:

- zero-cost conversion with native polars data types.
- Allows using regular polars operations directly on these new data types
- Full compatibility with streaming engine
- Mismatches, illegal operations etc. can be caught at plan time or expression creation time,
  rather than at runtime.

In addition, GeoPolars aims to be a modern GeoSpatial toolset, offering:

- CRS-aware by default, backed by PROJ
- Support for 3D (z) and measured quantities (m) for most operations
- (De)serialisation with GeoParquet

## Developer Documentation

If you are interested in the design paradigm and architecture used for this plugin,
See the `Docs` directory for more information about the internals of this plugin.

- [A high-level overview of the architecture of the plugin](./docs/source/development/architecture.md)
- [A document that describes what happens when you call an expression, at plan and execution time](./docs/source/development/typical-expression-flow.md).
  This is accompanied by [a diagram](./docs/source/development/expression-flow.dot).
- [What to consider when implementing new or editing existing expressions](./docs/source/development/expression-implementation-guide.md)
- [The views kernels take, in `geopolars-views`](./geopolars-views/README.md)
- [What to consider when writing kernels, in `geopolars-compute`](./geopolars-compute/README.md)
- [An overview of which expression is implemented at which *tier*](./docs/source/development/tier-per-expression.md)

## User guide

- [Getting started](./docs/source/user-guide/getting-started.md): the functional and the namespace API
- [Validity](./docs/source/user-guide/concepts/validity.md): what a valid geometry is, and how to check and repair it
- [Winding order](./docs/source/user-guide/concepts/winding-order.md)
- [CRS awareness](./docs/source/user-guide/concepts/crs.md)
- [User-defined functions](./docs/source/user-guide/udfs.md)

> [!TODO] the relative links in this README break on PyPI, which uses it as the package description.
Make them absolute once the docs are hosted.
