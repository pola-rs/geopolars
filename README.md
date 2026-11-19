<h1 align="center">GeoPolars</h1>

<div align="center">
  <a href="https://pypi.org/project/geopolars/">
    <img src="https://img.shields.io/pypi/v/geopolars.svg" alt="PyPI Latest Release"/>
  </a>
  <a href="https://github.com/pola-rs/geopolars/actions/workflows/test-python.yml">
    <img src="https://github.com/pola-rs/geopolars/actions/workflows/test-python.yml/badge.svg" alt="Python tests"/>
  </a>
  <a href="https://github.com/pola-rs/geopolars/actions/workflows/test-rust.yml">
    <img src="https://github.com/pola-rs/geopolars/actions/workflows/test-rust.yml/badge.svg" alt="Rust tests"/>
  </a>
  <a href="https://github.com/pola-rs/geopolars/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/geopolars.svg" alt="License"/>
  </a>
</div>

<p align="center">
  <b>Documentation</b>:
  <a href="https://github.com/pola-rs/geopolars/tree/main/geopolars/docs/source/reference">API reference</a>
  |
  <a href="https://github.com/pola-rs/geopolars/tree/main/docs/source/user-guide/geo">User guide</a>
</p>

## GeoPolars: geospatial data types and operations for Polars

GeoPolars adds first-class geospatial data types to Polars.
On top of being compatible with many existing Polars functionality,
these data types also come with a suite of geo-specific expressions.
They are regular Polars expressions,
so they work in lazy queries and the streaming engine like any other.

- **Native geometry types**: points, line strings, polygons and their multi forms,
  in 2D, with `z` and with `m`, stored as [GeoArrow](https://geoarrow.org/).
- **Checked at plan time**: a mismatched type or an unsupported operation
  is refused while the query is built, not halfway through running it.
- **CRS-aware**: every geometry column carries its CRS.
  Reprojection is backed by PROJ,
  and area and distance are measured on their ellipsoid for longitude/latitude.
- **GeoParquet**: read and write it with Polars' own Parquet readers and writers.

To learn more, read the [user guide](https://github.com/pola-rs/geopolars/tree/main/docs/source/user-guide/geo).

## GeoPolars in action

### Reading GeoParquet

GeoParquet keeps the encoding and the CRS of its geometry columns in the metadata of the file,
which `from_geoparquet` reads.
The CRS here is longitude/latitude, so `area` measures on the ellipsoid, in square metres:

```python
import polars as pl

import geopolars as gpl
from geopolars import MultiPolygonXY, geo

countries = (
    pl.scan_parquet("countries.parquet")
    .pipe(gpl.from_geoparquet, "countries.parquet")
    .select("name", geometry=geo.from_wkb("geometry", MultiPolygonXY))
    .with_columns(area_km2=geo.area("geometry") / 1e6)
    .sort("area_km2", descending=True)
    .collect()
)
```

### From longitude/latitude to GeoParquet

Build points from plain coordinate columns, reproject them,
and write a file that other GeoParquet readers (such as GeoPandas and GDAL) understand:

```python
cities = pl.scan_csv("cities.csv").select(
    "city",
    # EPSG:28992 is the Dutch national grid, in metres.
    geometry=geo.to_crs(geo.point("lon", "lat", crs="EPSG:4326"), "EPSG:28992"),
)
cities.sink_parquet("cities.parquet", metadata=gpl.geoparquet_metadata(cities))
```

Every operation is also in the `.geo` namespace:
`gpl.col("geometry").geo.area()` is the same as `geo.area("geometry")`.

## Installation

Install the latest GeoPolars version with:

```sh
pip install geopolars
```

GeoPolars is in alpha: expect breaking changes between releases.

<details>
<summary><b>Compile GeoPolars from source</b></summary>

GeoPolars builds against a pinned commit of Polars,
which `make install` checks out into `.polars` and builds as well.

1. Install the latest [Rust compiler](https://www.rust-lang.org/tools/install)
   and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install what the bundled PROJ and GEOS are built with:
   - a C++ compiler and CMake, for both.
   - SQLite, both the library and the `sqlite3` binary, for PROJ only.
   On Debian or Ubuntu: `sudo apt-get install build-essential cmake sqlite3 libsqlite3-dev`.
3. From the root of the repository, choose one of the following:
   - `make install`, an unoptimised build, with fast compile times.
   - `make install-release`, an optimised build, with long compile times.

Both set up a virtual environment in `.venv` with GeoPolars installed.
Run `make help` to see the other targets.

</details>

## Contributing

Want to contribute? Read our [contributing guide](https://github.com/pola-rs/geopolars/blob/main/docs/source/development/contributing/index.md)
and check the [issue tracker](https://github.com/pola-rs/geopolars/issues).

## License

GeoPolars is licensed under the [MIT License](https://github.com/pola-rs/geopolars/blob/main/LICENSE) (SPDX: `MIT`).
