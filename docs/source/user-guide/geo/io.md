# Reading and writing

GeoPolars reads and writes geometries with Polars' own readers and writers.
It only adds the steps around them:
giving geometry columns their data types after reading,
and describing them in the file when writing.

## GeoParquet

[GeoParquet](https://geoparquet.org/) is a Parquet file with a `geo` key in its metadata.
That key says which columns hold geometries, how they are encoded, and in which [CRS](concepts/crs.md).
GeoPandas, GDAL, DuckDB and many other tools read and write it.

GeoParquet is supported through Polars itself.
It is experimental, and only switched on by environment variables.
Use `POLARS_GEOPARQUET_READ=1` to read the `geo` metadata,
and `POLARS_GEOPARQUET_WRITE=1.0` or `1.1` to write it.
Without them, files from other tools read as plain structs and binary,
and written files carry no `geo` metadata.

### Writing

Polars writes geometry columns to Parquet as they are.
Other tools only see them as geometries, though, if the file also has the `geo` key.
`geoparquet_metadata` gives you that key.
Pass it to `write_parquet` or `sink_parquet`, next to any metadata of your own:

{{code_block('user-guide/geo/io','write',['geo.point','geoparquet_metadata','write_parquet'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:write"
```

Not everything GeoPolars can hold fits into GeoParquet 1.1.
It cannot store [m values](concepts/geometry-types.md#dimensions), nor WKT,
and it needs a CRS that can be written as [PROJJSON](https://proj.org/specifications/projjson.html),
the JSON form of a CRS.
Its edges have to be planar or spherical.
`geoparquet_metadata` refuses such a column.
It only needs the schema to do so, so it refuses before anything is written:

{{code_block('user-guide/geo/io','refuse',['geoparquet_metadata'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:refuse"
```

### Reading

Polars reads a file it wrote itself with all its data types intact.
A file from another tool only has the `geo` key to go on.
`from_geoparquet` reads that key, and gives each geometry column its data type.
It reads the key from the file itself, so you pass it the path as well:

{{code_block('user-guide/geo/io','read',['scan_parquet','from_geoparquet'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:read"
```

This works on a `LazyFrame` and on a `DataFrame` alike,
and with the same paths, globs and cloud storage as `scan_parquet`.
A plain Parquet file, without the `geo` key, is left as it is.
By default, `from_geoparquet` checks that every file describes its geometries in the same way,
and refuses a mix of files with and without the key.
`check_every_file=False` only reads the key of the first one, which is faster for many files.

GeoParquet stores geometries in one of two ways.
A *native* column comes in as its geometry type, such as `PointXY`, and is ready to use.
A [WKB](#wkt-and-wkb) column comes in as `Wkb`.
Decode it with `from_wkb`, into the type you expect.
Below, we first write the cities as WKB, and then read them back:

{{code_block('user-guide/geo/io','wkb',['geo.to_wkb','geo.from_wkb'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:wkb"
```

!!! warning "The CRS comes back as PROJJSON"

    GeoParquet writes every CRS as PROJJSON, and `from_geoparquet` labels the column with it.
    A column you declared as `"EPSG:4326"` therefore comes back with a different label.
    (The header of the DataFrame shortens it to `EPSG:4326`, but the label is the full PROJJSON.)
    As GeoPolars [compares CRSs by how they are written](concepts/crs.md#combining-columns),
    it no longer combines with columns that still say `"EPSG:4326"`.

If you know both mean the same CRS, relabel the column with `set_crs(..., force=True)`:

{{code_block('user-guide/geo/io','crs',['geo.set_crs','geo.distance'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:crs"
```

See [GeoParquet](concepts/crs.md#geoparquet) on the CRS page for how a column without a CRS is read.

### Bounding boxes

A GeoParquet file can store a bounding box next to each geometry,
which readers can use to skip parts of the file they do not need.
`with_covering` adds a box column `<name>_bbox` for every geometry column,
and `geoparquet_metadata(..., covering=True)` describes it in the file.
A multi geometry gets one box around all of its parts.
A `Wkb` column cannot be bounded, so decode it with `from_wkb` first:

{{code_block('user-guide/geo/io','covering',['with_covering','geoparquet_metadata','sink_parquet'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:covering"
```

When you read the file back, the box column comes in as a [box](concepts/geometry-types.md#boxes).

### Known limitations

- A `bbox` covering is only read as a box if its fields are `Float64`,
  in the order `xmin, ymin, [zmin,] xmax, ymax, [zmax]`.
  Any other covering (such as the one in the GeoParquet spec's own example file)
  is read as a plain struct, with a warning.

### Repairing what Polars does not read

When Polars cannot honour the `geo` metadata of a column,
it does not refuse the file.
It reads the column as plain Polars data instead, and warns.
This happens with an unknown encoding or version,
storage that does not match the encoding, or a malformed covering.

The plain column has lost its CRS, so first read it from the file yourself.
In GeoParquet, a missing `crs` means `OGC:CRS84`, and `null` means unknown.

```python
import json

column = json.loads(pl.read_parquet_metadata(path)["geo"])["columns"]["geometry"]
crs = column.get("crs", "OGC:CRS84")
crs = json.dumps(crs) if isinstance(crs, dict) else crs
```

Then turn the column into a geometry:

- Storage that already fits a GeoPolars type: `pl.col("geometry").ext.to(PointXY(crs=crs))`.
- Other storage: the matching constructor or decoder, such as `geo.from_wkt("geometry", PointXY, crs=crs)`.
- A covering with all its bounds:
  `geo.box(bound("xmin"), bound("ymin"), bound("xmax"), bound("ymax"), crs=crs)`,
  with `bound = pl.col("bbox").struct.field`.
- A covering with too few bounds: compute it again with `geo.bounds("geometry")`.

GeoPandas writes the covering of a null geometry as a box with every bound null,
rather than as a null box.
Polars reads it as it is, which breaks GeoArrow's rule on nulls.
`geo.null_out_invalid_geoarrow("bbox")` makes those boxes null.

## WKT and WKB

WKT (well-known text) and WKB (well-known binary) are the most common ways to write a single geometry.
You will find WKT in CSV files, and WKB in databases and older Parquet files.

`from_wkt` and `from_wkb` decode a column of them.
A WKT or WKB column can mix kinds of geometry from row to row, but a geometry column cannot.
So you pass the [geometry type](concepts/geometry-types.md#one-kind-per-column) to decode into.
You can also pass the CRS the coordinates are in:

{{code_block('user-guide/geo/io','wkt',['read_csv','geo.from_wkt','MultiPolygonXY'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:wkt"
```

`to_wkt` and `to_wkb` go the other way.
They give a `Wkt` or a `Wkb` column, which keeps the CRS.
WKT is written with every digit, so it reads back exactly.
WKB is written as ISO WKB, which some older readers do not accept.

## Other formats

GeoPolars does not read GeoJSON, shapefiles, or other GDAL formats yet.
Read them with [GeoPandas](https://geopandas.org/) instead, and convert the result
(see [coming from GeoPandas](migration/geopandas.md#converting-between-geopandas-and-geopolars)).
