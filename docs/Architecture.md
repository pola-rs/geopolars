# GeoPolars Architecture

This document gives a rough overview of the architecture of the GeoPolars plugin.
I hope this helps you understand the outline a little better, and tells you who's who,
and where to find certain functionality!
It might be useful to also read [Typical program flow],
which describes the typical flow of an expression evaluation.

## A Polars plugin

GeoPolars is a [Polars plugin](https://docs.pola.rs/user-guide/plugins/):
a Python package with a compiled Rust library (as `cdylib`).
The two halves each have their own job:

- **Python** (`geopolars/`) is what users interact with.
  It defines the dtypes, builds the expressions,
  and decides at plan time which implementation each geometry gets.
- **Rust** (`src/`) holds the kernels that plain Polars expressions can't express.
  Polars loads the library itself and looks up each plugin function by itself.

The Python side points Polars at the library through `LIB` in `_utils.py`,
which is the package directory.

## GeoArrow data types

Geometries are stored as [GeoArrow](https://geoarrow.org/),
using [Polars' Arrow extension types](https://docs.pola.rs/api/python/stable/reference/api/polars.datatypes.Extension.html).
An extension type consists of a name (`geoarrow.point`, `geoarrow.polygon`, ...),
an ordinary Polars storage dtype, and optional metadata (where the CRS lives).
The storage is a struct of separate `xy[/z][/m]` `f64` fields (referred to as *coordinates*),
wrapped in a `List` per level of nesting (a point has 0, a linestring 1, a polygon 2).

| What | Python | Rust |
| - | - | - |
| List of geometries | `GEOMETRY_DTYPES` in `datatypes/registry.py` | `Kind::ALL` in `geopolars-views/src/kind.rs` |
| Common base type | `GeoArrowType` in `datatypes/base.py` | `Geo` in `src/geoarrow/geo.rs` |
| One geometry | `datatypes/point.py`, `polygon.py`, ... | a `Kind` variant |
| Dimension (`xy`, `xyzm`) | `datatypes/dimension.py` | `src/geoarrow/dimension.rs` |
| CRS metadata | goes through `extension_metadata` (Rust) | `src/geoarrow/crs.rs` |

Two things work differently on the two sides:

- In Python, each geometry *and dimension* is its own class (`PointXY`, `PolygonXYZM`),
  so users and type checkers can tell them apart.
  In Rust, there is a single `Geo { kind, dim }` and code matches on its enums,
  the same way Polars itself dispatches on dtypes.
- Metadata JSON is only ever handled in Rust to keep a single source of truth.
  Python calls `extension_metadata(crs=...)` rather than writing the JSON itself.
  The reason for this is twofold:
  It stops operations on the metadata from drifting (lowering maintenance burden),
  and it ensures that metadata processed by both sides can be compared.
  Dtypes compare their metadata as strings,
  so if the two sides serialised the same CRS differently
  you'd get two dtypes that don't compare equal.
  In Rust, metadata is held parsed as an `ExtensionMetadata`, behind an `Arc`,
  so it compares by content and is written back out as compact JSON.

`geoarrow.box` (`datatypes/box.py`, `src/geoarrow/bbox.rs`), which represents a bounding box,
is registered on both sides too, but it is not one of these geometries:
its storage is a flat struct of bounds (`xmin, ymin, ..., xmax, ymax, ...`),
so, like WKB and WKT, it is not a `Kind`, and it is not accepted by most operations.

Rust expressions never name a concrete geometry.
They call `geoarrow::describe(dtype)` to get a `GeoColumn` (kind and dimension),
then either ignore it (`coords::map_coords`) or match exhaustively on `Kind`.
That way, adding a geometry only touches `src/geoarrow/` and doesn't affect `src/expr/`.

### Extension registries

There are two extension-type registries, one in the host `polars` wheel and one inside our `cdylib`,
so **both Python and Rust have to register every type, and the names have to agree**.
Importing `geopolars` handles both:
`datatypes/registry.py` imports the compiled module,
which runs `PyInit_geopolars` and registers the Rust side,
then calls `pl.register_extension_type` for the Python side.

## Expressions

Expressions are the operations on these types.
Each operation is written once, as a function in the **functional API** (`geopolars/geo/`),
which is available as `from geopolars import geo` and then `geo.area("parcel")`.
The **namespace API** (`geopolars/expr/geo.py`, `gpl.col("parcel").geo.area()`)
is a thin wrapper in which every method just forwards to the functional version.
New functionality is added in `geo/`, plus a one-line method in the namespace.

Because of this, dependencies only go one way.
Nothing in `datatypes` imports from `geo`, and nothing in `geo` imports from `expr`.

```text
expr (namespaces) →  geo (functional) →  datatypes.
```

On the Rust side, `src/expr/` mirrors `geo/` module by module (`area.rs`, `affine.rs`, `crs.rs`, ...),
and `src/geoarrow/` is the shared glue: dtypes and dispatch.
`src/proj.rs` asks PROJ about a CRS.
Reading the Arrow storage in a kernel goes through the views of `geopolars-views`.

## Type checker

`register_expr_namespace` patches `.geo` onto `pl.Expr` at runtime, which a checker can't see.
To fix this, we make `gpl.col` / `as_plugin` return a `PluginExpr` subclass that declares it.
This is the purpose of `expr/plugin.py`.
