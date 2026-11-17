# GEOS kernel implementation guide

This document extends the [Kernel implementation guide](<Kernel implementation guide.md>),
for kernels in `geos` that convert to GEOS and call it.

## Shapes

A GEOS kernel has one of the shapes below.
The sketches show the target.
Each names the module closest to it today, and how that module still differs.

### Unary, giving geometries

```text
view -> GeosGeometries -> (GEOS, row by row, in place) -> GeosGeometries -> view
```

First, `to_geos_rows` converts the view.
A private function then replaces each `GeosRow::Geometry` with what GEOS makes of it,
and an `into_*_array` conversion gives the view of the output kind.
So something that looks like this:

```rust
pub fn simplify_polygons(
    array: &PolygonArray,
    tolerance: f64,
    preserve_topology: bool,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<PolygonArray> {
    let geometries = array.to_geos_rows(on_row_error, early_coord_check)?;
    simplify_via_geos(geometries, tolerance, preserve_topology, on_row_error)?
        .into_polygon_array(array.coords.dim().has_z(), on_row_error, …)
}

fn simplify_via_geos(
    mut geometries: GeosGeometries,
    tolerance: f64,
    preserve_topology: bool,
    on_row_error: OnRowError,
) -> GeoComputeResult<GeosGeometries> {
    with_const_flags!(
        PRESERVE_TOPOLOGY = preserve_topology;
        simplify_via_geos_with::<PRESERVE_TOPOLOGY>(&mut geometries.geometries, tolerance, on_row_error)
    )?;
    Ok(geometries)
}

fn simplify_via_geos_with<const PRESERVE_TOPOLOGY: bool>(
    geometries: &mut GeosRows,
    tolerance: f64,
    on_row_error: OnRowError,
) -> GeoComputeResult<()> {
    geometries.replace_geometries(on_row_error, |_, input| {
        … // GEOS on `input`
    })
}
```

### Binary

Both sides come in as a `&GeometryArray`
(so both columns can have a different type, but all items in the columns are all the same type).
`GeosPairs` converts them and gives the pairs of `GeosRow`s.
It is also responsible for broadcasting a side of a single row against the other.
A function per pair gives the result of that pair.

```rust
pub fn topologically_equal(
    a: &GeometryArray,
    b: &GeometryArray,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<BooleanArray> {
    let pairs = GeosPairs::new(a, b, on_row_error, early_coord_check)?;
    let tested = pairs
        .iter()
        .map(|(a, b)| row_or_null(rows_are_topologically_equal(a, b), on_row_error))
        .collect::<GeoComputeResult<Vec<_>>>()?;
    Ok(optional_bool_array(&tested))
}
```

A binary operation with several kinds of output keeps the pairs in a struct,
with one `into_*` method per output kind (see `OverlayPairs` in `geos/overlay.rs`).

### Predicates on the parts

Should create a bool bitmap.
Can use the same mechanism as any of the other GEOS things, so that they can theoretically be chained together later.

## One container for GEOS geometries

The geometries in a column of every GEOS kernel go in one `GeosGeometries`.
The kind of the output lives in the `into_*_array` conversion,
not in a container of its own.

A `GeosRow` is `Empty` or a `Geometry` (null is described with the validity bitmask.
We can just plonk `Empty` there, it should not matter).
Two more are planned, `LeftUnchanged` and `RightUnchanged`,
for a binary operation that gives one of its inputs back as it is.

## Trivial rows never reach GEOS

Converting a geometry to GEOS and calling GEOS on it is expensive.
A kernel that can tell the result of a row without GEOS gives it before converting.
Null rows are unset in the validity of `GeosRows`, and empty rows become `GeosRow::Empty`.
These are much cheaper to create,
and let us switch what we do without inspecting the GEOS data.

In a binary operation, a null or an empty on either side should short-circuits the pair,
foregoing the need to convert its partner in the other column.

## Coordinate checks and GEOS errors

- A GEOS kernel defaults to `early_coord_check="finite_and_empty"`.
  The check runs during the conversion to GEOS.
- `"finite"` lets an empty point inside a geometry through to GEOS.
  GEOS is allowed to deal with the empty point as it normally would,
  which in most cases means dropping it.
  An empty point on its own passes with either, as GEOS supports it.
- If the check can be turned off,
  the module documents what GEOS does with non-finite input then,
  as `geos/convex_hull.rs` does.
- An error from GEOS goes through `on_row_error` (e.g. with `row_or_null`),
  never straight to a null.

## Orientation of new rings

GEOS operations that build new rings always give clockwise shells and counter-clockwise holes,
whatever the orientation of the input. This held for (among others):

- intersection, union, difference and unary_union
- buffer, including buffer(0) and Point.buffer
- normalize
- make_valid on invalid input (the bowtie came back as two polygons, both with clockwise shells)

For these, the output already has a known orientation,
so `new_ring_orientation="cw"` with `new_holes="opposite"` is a no-op, and can be skipped.

Operations that leave the rings alone keep the orientation of the input though:

- simplify returned a counter-clockwise shell with a counter-clockwise hole unchanged.
- make_valid on input that is already valid returns it as is, even when the hole has the wrong orientation.
  A valid polygon doesn't need any particular orientation, so this case passes straight through.

Test it before assuming anything.
