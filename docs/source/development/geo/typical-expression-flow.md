
# Typical flow of a plugin expression evaluation

This section describes what happens when you add a typical expression
(in this case, `geo.area`) to a query.
This should give you a good idea of how the plugin works.
This is a written description of `docs/source/development/geo/expression-flow.dot`
(which you can render with `dot -Tsvg docs/source/development/geo/expression-flow.dot -o flow.svg`).

At two points, the paths diverge depending on which tier the operation is implemented at.
The numbering here is the same as before.
Blue steps run in our Python package, orange ones in our library.
Every red edge crosses the FFI.
What happens inside Polars itself (in-between) is left out.

## Adding an expression

( The top half of `docs/source/development/geo/expression-flow.dot` )

This code mostly lives in the Python part of GeoPolars.

There are two ways in:

- The functional API (A2), with a column name, an expression or a Series.
- The namespace (`ExprGeoNameSpace.area`) which simply forwards (A1) to the functional API directly.

From there, the operation goes one of two ways,
depending on whether it needs to be aware of type information (A3).
The input's dtype isn't known yet, so nothing can be checked against it.
This matters for C-tier functions,
but also for some other functions which switch their approach based on the geometry or metadata.
An example of such a function would be `area`, which is B-tier for a polygon with a CRS
(which is stored in metadata), but A-tier for one without.
These need to choose at plan-time, so they can decide when this extra information is available.
They do this with a callback, which is called in P4.

For some expressions, you would always use an A-tier or B-tier implementation (A4).
These can use the simple path.

## With a plan-time callback

`on_geometry` turns the input into an expression (A5),
where a name becomes `pl.col`, a Series `pl.lit`.
It then hands the operation's callback (unique for every operation) to `pipe_with_dtype`
(through its own wrapper, `resolved` ) (A6).
This yields an ordinary `pl.Expr` that carries the callback that can run at plan time.
It picks up again at P1.

### The simple path

These do not need to know the exact data type here.
The rust side will do the switching inside of their A-tier or B-tier kernel.

For A-tier, the arguments are packed into `kwargs` (A7),
which the rust side can later deserialize into a struct like `AffineMatrix`.
B-tier works basically the same way here, with kwargs for what the external library needs (A9),
(e.g. the target CRS of `to_crs`).

( For some complex ops, a dtype the operation is asked to produce can't cross as a Python object,
so `from_wkb` flattens it into kind, dimension and CRS (`_decode_kwargs`).)

`register_plugin_function` then names the kernel by its symbol in `LIB` (A8).
That records the name, so the appropriate rust function can be called later in P8.

The forwarding kernel is named like any other (A10), and picks up again at P13.

## Planning an expression

( The bottom half of `docs/source/development/geo/expression-flow.dot` )

Type level checking is implemented on this level as much as possible,
so that they can raise while the plan is being built, rather than during execution.

When the query is collected, Polars resolves the input's dtype and calls back into our code.
An A-tier or B-tier expression that was directly chosen without a `pipe_with_dtype`
(from A8 or A10) does not use a callback, and goes straight to P8 or P13.

The other path is one that requires a callback.
A C-tier expression (or one that resolves based on metadata) from A6 starts at P1.

### Callbacks with geometry and metadata

Polars hands the dtype back (across the FFI) with the python callback.
This will be one of our registered new types, such as `PolygonType` (P1).
It can then choose a concrete subclass (e.g. `PolygonXY`),
based on the dimension using its implementation for `ext_from_params`.
Metadata is left as-is for now.
Storage we don't support, such as interleaved coordinates, raises here.
`_geometry_of` then refuses anything that isn't one of our geometries,
such as a `Wkb` column or a plain struct (P2).

If the choice of implementation depends on the metadata, the callback calls into Rust (P3):
`_declares_crs` calls `declares_crs` in our library directly through pyo3, not through Polars.
Right now we only check whether a CRS exists, but we might expand on that.
We could return the entire CRS back over to Python here.

Finally, the callback chooses the implementation (P4).
`on_geometry` gives whatever it returns the input's name.

### C-tier: Polars expressions

The callback builds the result out of the storage in plain Polars expressions (P5).
`.ext.storage()` drops the extension type,
and the class's `_nesting`, `_rings` and `_dimension` say how many `List` layers there are,
and which coordinate fields.
A geometry result gets its dtype back with `.ext.to(...)` (P6).
Along with it, input's metadata is carried as an unparsed string using `_with_metadata_of`,
Some plain results (`f64`) need nothing, so the CRS can be discarded then.
Polars can then read the dtype off the expression as its name, storage and metadata.

After this, our code is done with the expression.
The plan only holds Polars nodes, which can be optimised like any other.

### A-tier: native kernel

We can get here either directly from a registered kernel that was lowered immediately (A8),
or because a callback at planning time lowered to this (P4).

Which function is run depends on what was registered as its `register_plugin_function` call (P7).
Polars then calls `_polars_plugin_field_<name>` in our library over the C ABI (P8),
passing the input fields as Arrow C schemas and the pickled kwargs.
Importing a field looks its name up in our library's own registry.
`GeoFactory` can now parse the field.
This reads:

- The `Kind` from the name
- the `GeoDimension` from the storage
- the `ExtensionMetadata` from its metadata JSON.

That `GeoFactory` parses that into a `Geo`,
or an `Unsupported` if the storage or metadata didn't parse (the factory cannot return an error).
`describe` turns the dtype into a `GeoColumn` (P9),
and refuses an `Unsupported` or anything that isn't a geometry with a `SchemaMismatch`
(`from_wkb`, whose input is encoded, uses `describe_encoded` instead).
Whatever output type function (in Rust) this uses determines the output field type (P10).

- `same_geometry` hands back the input's field,
- `to_wkb` gives a `Wkb` carrying the same metadata.
- and any other function we can call to return that type.

The field goes back over the C ABI.
A dtype registered from Python, such as a geometry or `Wkb`,
is rebuilt through our `ext_from_params` in Python as in P1 (P11),
and a plain one (`f64`) goes straight back.
The plan now holds a node for our kernel, with a known output dtype.

Ready to run!

### B-tier: forward to external

Steps P12 to P16 are the same as P7 to P11, but now for a forwarding function.
