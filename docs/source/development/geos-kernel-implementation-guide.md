# GEOS kernel implementation guide

This document extends the [Kernel implementation guide](kernel-implementation-guide.md),
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
with `replace_geometries`.
It also takes the orientation of what GEOS gives
(see [Orientation of new rings](#orientation-of-new-rings)),
so that the column's orientation always matches its rows.
Every other row stays as it is,
and a row GEOS fails on under `on_row_error="null"` becomes `Null`.
Then, an `into_*_array` conversion gives the view of the output kind.
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
        simplify_via_geos_with::<PRESERVE_TOPOLOGY>(&mut geometries, tolerance, on_row_error)
    )?;
    Ok(geometries)
}

fn simplify_via_geos_with<const PRESERVE_TOPOLOGY: bool>(
    geometries: &mut GeosGeometries,
    tolerance: f64,
    on_row_error: OnRowError,
) -> GeoComputeResult<()> {
    // Both simplifiers keep the rings of the input, whose orientation is unknown.
    geometries.replace_geometries(GeometryOrientation::UNKNOWN, on_row_error, |_, input| {
        … // GEOS on `input`
    })
}
```

### Binary

Both sides come in as a `&GeometryArray`, so both columns can have a different type,
but all items in the columns are all the same type.
The operation implements `PairOperation`,
which says what a pair of two geometries makes,
but also what to do when either of the two sides (or both) is empty.
`GeosPairs` does the plumbing here:
it converts only the rows the operation needs
(see [Trivial rows never reach GEOS](#trivial-rows-never-reach-geos)),
broadcasts a side of a single row against the other,
and gives null for a pair with a null row.
A row that failed to convert under `on_row_error="null"` only nulls the pairs that need it.
Paired with an empty row whose result needs no partner (`WhenEmpty::Gives`),
it still gives that result.
An operation therefore never sees a row that was not converted,
and does not handle nulls or `on_row_error` itself.

```rust
struct TopologicallyEqual;

impl PairOperation for TopologicallyEqual {
    type Output = bool;

    fn when_both_empty(&self) -> bool {
        true
    }

    fn when_a_empty(&self) -> WhenEmpty<Self> {
        // `Gives` needs no partner, so the partner is never converted.
        // `WithPartner` would get it as a GEOS geometry.
        WhenEmpty::Gives(|| false)
    }

    fn when_b_empty(&self) -> WhenEmpty<Self> {
        WhenEmpty::Gives(|| false)
    }

    fn on_geometries(&self, a: &Geometry, b: &Geometry) -> GeoComputeResult<bool> {
        Ok(a.equals(b)?)
    }
}

pub fn topologically_equal(
    a: &GeometryArray,
    b: &GeometryArray,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
) -> GeoComputeResult<BooleanArray> {
    let pairs = GeosPairs::new(a, b, &TopologicallyEqual, on_row_error, early_coord_check)?;
    let tested = pairs
        .evaluate(&TopologicallyEqual, on_row_error)
        .collect::<GeoComputeResult<Vec<_>>>()?;
    Ok(optional_bool_array(&tested))
}
```

A binary operation with several kinds of output keeps the pairs in a struct,
with one `into_*` method per output kind (see `OverlayPairs` in `geos/overlay.rs`).
Its operation gives the result before it is narrowed down to the output kind,
as the empty rows decide what is converted before that kind is known.
Each `into_*` method then narrows the result of every pair.

### Predicates on the parts

Should create a bool bitmap.
Can use the same mechanism as any of the other GEOS things, so that they can theoretically be chained together later.

## One container for GEOS geometries

The geometries in a column of every GEOS kernel go in one `GeosGeometries`,
which is made up of `GeosRow`s.
These can hold several variants so that empty and invalid cases can be handled consistently:

| `GeosRow` | The row | Made by | In the output |
| --- | --- | --- | --- |
| `Geometry` | holds a GEOS geometry | the conversion, or the kernel | that geometry |
| `Empty` | has no vertices (see [Trivial rows never reach GEOS](#trivial-rows-never-reach-geos)) | the conversion, or the kernel | an empty geometry |
| `Null` | is null | the conversion, or the kernel | null |
| `Failed` | failed to convert under `on_row_error="null"` | the conversion | null |
| `Skipped` | was not converted, as its partner decides result | `GeosPairs` | never gets there |

A kernel only ever makes `Geometry`, `Empty` or `Null` itself.
`Failed` and `Skipped` only exist in intermediate stages.
In a binary operation, having a null or an empty row on either side can decide the result
without even needing to look at the other side.
Take for example an intersection operation:
If one side is empty, then we don't even need to look at the other (valid) side,
as we already know that the output will be empty too.
In such a case, the other side will be set to `Skipped`.
`Failed` holds the `on_row_error="null"` verdict back until a result needs the row,
so that a binary operation can null only the pairs that need it.
A result row that GEOS fails on has had its verdict, so it is `Null`, not `Failed`.
`Null` also covers a result that is null by definition,
such as the convex hull of a single coordinate.

## Trivial rows never reach GEOS

Converting a geometry to GEOS and calling GEOS on it is expensive.
For trivial cases, it is much cheaper to bypass it if we already know the answer.
The conversion already does this for null rows and rows that are empty (without vertices, or no parts).
which become `GeosRow::Null` and `GeosRow::Empty` without going to GEOS.
These rows are much cheaper to create,
and let us switch what we do without inspecting the GEOS data.

## Coordinate checks and GEOS errors

- A GEOS kernel defaults to `early_coord_check="finite_and_empty"`.
  The check runs during the conversion to GEOS.
- `"finite"` lets an empty point inside a geometry through to GEOS.
  What GEOS then does is described under `early_coord_check` in [Common arguments](../../../geopolars/docs/source/reference/common_arguments.rst).
- If the check can be turned off,
  the module documents what GEOS does with non-finite input then,
  as `geos/convex_hull.rs` does.
- An error from GEOS goes through `on_row_error` (e.g. with `row_or_null`),
  never straight to a null.

## Orientation of new rings

Which way GEOS runs the rings it builds is described under [New rings](../user-guide/geo/concepts/winding-order.md#new-rings) in the user guide.

The takeaway is that it is not consistent,
which is why every operation must therefore set the `orientation` of its output `GeosGeometries`.
Every row of the column runs in that direction,
unless its `GeosRow::Geometry::maybe_wrong_orientation` is set.

When an orientation is asked for, a row without the flag is reversed based on the column's `orientation` alone,
without its rings being checked.
A row that may run otherwise, such as one passed through unchanged, must therefore have the flag.
The rings of a row with the flag are checked when they are copied out of GEOS,
and only reversed where needed.
In a column whose `orientation` is even partly `Unknown`, every row is checked this way.
With `new_ring_orientation="keep"`, you indicate that you don't care about the orientation.
This bypasses checking and reversing, which is faster.

Test it before assuming anything!
