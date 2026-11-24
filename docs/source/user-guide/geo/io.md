# Reading and writing

GeoPolars reads and writes geometries with Polars' own readers and writers,
which handle GeoParquet natively.
GeoPolars only adds some utilities around them,
such as decoding WKB and adding bounding boxes.

## GeoParquet

[GeoParquet](https://geoparquet.org/) is a Parquet file with a `geo` key in its metadata.
That key says which columns hold geometries, how they are encoded, and in which [CRS](concepts/crs.md).
GeoPandas, GDAL, DuckDB and many other tools read and write it.

Polars' GeoParquet support is still experimental, and only switched on by two environment variables.
`POLARS_GEOPARQUET_READ=1` reads the `geo` key,
and `POLARS_GEOPARQUET_WRITE=1.1` (or `1.0`) writes it.
Polars reads the variables at every read and write,
so setting them through `os.environ` within Python works too.

### Writing

Polars writes geometry columns to Parquet in the struct format they are stored in.
The arrow type is also written, which is useful for a Polars roundtrip,
but not something you can rely on for other consumers.
Only if the `geo` key is set in the metadata can other consumers know that this is GeoParquet.
Setting the `POLARS_GEOPARQUET_WRITE=1.1` environment variable makes it so
that `write_parquet` and `sink_parquet` add that key, and describe every geometry column in it.
GeoParquet 1.0 only holds WKB, so `POLARS_GEOPARQUET_WRITE=1.0` refuses a native geometry column.

GeoParquet is a little bit more limited in what it can store than what GeoPolars can represent.
GeoParquet 1.1 cannot store [m values](concepts/geometry-types.md#dimensions), nor WKT.
Its edges also have to be planar or spherical.

To make sure that information is not silently lost,
Polars refuses such a column before anything is written:

{{code_block('user-guide/geo/io','refuse',['write_parquet'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:refuse"
```

### Reading

Polars reads a file it wrote itself with all its data types intact due to the stored Arrow data.
A file from another tool only has the `geo` key to go on.
The environment variable `POLARS_GEOPARQUET_READ=1` makes `read_parquet` and `scan_parquet`
check for that metadata key.

{{code_block('user-guide/geo/io','read',['scan_parquet'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:read"
```

GeoParquet stores geometries in one of two ways.
A *native* column comes in as its geometry type, such as `PointXY`, and is ready to use.
A [WKB](#wkt-and-wkb) column comes in as `Wkb`.
Decode it with `from_wkb`, into the type you expect.
Below, we first write the cities as WKB, and then read them back:

{{code_block('user-guide/geo/io','wkb',['geo.to_wkb','geo.from_wkb'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:wkb"
```

### CRS

GeoParquet can take a CRS written as [PROJJSON](https://proj.org/specifications/projjson.html)
(the JSON form of a CRS).
Specifying 'no crs' is done by explicitly setting the crs key to `null`.
If the key is missing, it is assumed to be in `"OGC:CRS84"`.
A string-CRS such as `"EPSG:4326"` is valid in GeoPolars (as in GeoArrow), but not for GeoParquet.
You'll have to convert it to its PROJJSON equivalent using `crs_to_projjson`.

{{code_block('user-guide/geo/io','write',['geo.point','geo.crs_to_projjson','write_parquet'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:write"
```

The same holds for reading data from GeoParquet that has a CRS.
You might now have your own column with the (string) label `"EPSG:4326"`,
and one you read from GeoParquet with the PROJJSON representation of that same CRS.
As GeoPolars [compares CRSs by how they are written](concepts/crs.md#combining-columns),
it no longer combines with columns that still say `"EPSG:4326"`.
If you know both mean the same CRS, relabel the column with `set_crs(..., force=True)`:

{{code_block('user-guide/geo/io','crs',['geo.set_crs','geo.distance'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:crs"
```

### Bounding boxes

A GeoParquet file can store two kinds of bounding boxes,
which readers can use to skip parts of the file they do not need.
A per-row bounding box sits next to each geometry (a *covering*),
and a per-column bounding box holds the extent of all the data in that column.

#### Per-column

Polars fills in the per-column bounding box from the statistics it writes into the file.
It does this for every native geometry column, but a `Wkb` column gets none.

#### Covering

Polars matches a covering column to its geometry column *by name*.
`with_covering` adds one for every geometry column, with the name Polars expects.
See its API reference for details.

{{code_block('user-guide/geo/io','covering',['with_covering','sink_parquet'])}}

```python exec="on" result="text" session="user-guide/geo/io"
--8<-- "python/user-guide/geo/io.py:covering"
```

When you read the file back, the covering comes in as a [box](concepts/geometry-types.md#boxes).
Polars only reads it as a box if its bounds are `Float64`,
in the order `xmin, ymin, [zmin,] xmax, ymax, [zmax]`.
It reads any other covering as a plain struct, and warns.

Polars refuses to write a box column that is not the covering of a geometry column,
rather than silently write it as a plain struct.
To write it as a plain struct anyway, convert it with `.ext.storage()` first.

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
