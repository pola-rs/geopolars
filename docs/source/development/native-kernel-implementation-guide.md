# Native kernel implementation guide

This document extends the [Kernel implementation guide](kernel-implementation-guide.md),
for kernels in `native` that compute their result themselves.

## Shapes

A native kernel has one of the shapes below.

### Measuring or testing each row

One value per row, such as an area, a count or a `bool`.
A private `#[inline(always)]` function computes it for one item,
and a loop runs it over the rows through the helpers in `per_row`
(`per_row`, `per_row_count`, `per_row_bool`, `per_row_optional_bool`, …),
which take care of the null rows.

```rust
#[inline(always)]
fn count_polygon_vertices<const INCLUDE_EMPTY: bool>(polygon: &ScalarPolygon) -> u32 { … }

pub fn polygon_coordinate_counts(
    array: &Untrimmed<PolygonArray>,
    include_empty: bool,
) -> PrimitiveArray<u32> {
    with_const_flags!(
        INCLUDE_EMPTY = include_empty;
        polygon_coordinate_counts_with::<INCLUDE_EMPTY>(array)
    )
}

fn polygon_coordinate_counts_with<const INCLUDE_EMPTY: bool>(
    array: &Untrimmed<PolygonArray>,
) -> PrimitiveArray<u32> {
    per_row_count(array.rows(), |i| {
        count_polygon_vertices::<INCLUDE_EMPTY>(&array.get(i))
    })
}
```

### Modifying the underlying coordinates

Stuff like clamping, snapping or reprojecting.
For this, a private function rewrites all the coordinates at once (using a bulk operation if possible).
As it walks every coordinate, it takes trimmed views.

```rust
fn clamped(coords: &ValidCoords, bounds: &Bounds) -> ValidCoords { … }

pub fn polygons_with_clamped_coords(array: &PolygonArray, bounds: &Bounds) -> PolygonArray {
    PolygonArray::from_parts(
        array.rows.clone(),
        array.rings.clone(),
        clamped(&array.coords, bounds),
    )
}

pub fn points_with_clamped_coords(array: &PointArray, bounds: &Bounds) -> PointArray {
    PointArray::new(Coords::from_valid(
        clamped(array.coords.ignoring_validity(), bounds),
        array.coords.validity().cloned(),
    ))
}
```

### Marking coordinates, then folding them up to the rows

One `bool` per row that depends on its coordinates, such as whether it has repeated points.
One pass marks the coordinates (or segments) in a bitmap, without looking at the rows.
`valid_rows_passing` in `passing` then folds that bitmap up, level by level, to the rows.

```rust
pub fn polygons_have_repeats_in_succession(
    array: &PolygonArray,
    include_z: bool,
    include_m: bool,
) -> BooleanArray {
    with_const_flags!(
        INCLUDE_Z = include_z, INCLUDE_M = include_m;
        polygons_have_repeats_in_succession_with::<INCLUDE_Z, INCLUDE_M>(array)
    )
}

fn polygons_have_repeats_in_succession_with<const INCLUDE_Z: bool, const INCLUDE_M: bool>(
    array: &PolygonArray,
) -> BooleanArray {
    let repeated: Bitmap = repeats_in_succession::<INCLUDE_Z, INCLUDE_M>(&array.rings, &array.coords);
    let passing = valid_rows_passing(&array.rows, &[&array.rings], Some(!&repeated));
    …
}
```

## Points and nested kinds

A kernel that works on both a nested geometry type
and a flat one (points) writes the work once,
as an `#[inline(always)]` function on `&ValidCoords` and a range of coordinates.
Note that `ValidCoords` does not have null information. That is added afterwards.
Each public function then calls it the way its view needs:

For **Points**, this is as `array.coords.ignoring_validity()`,
after which the output gets `array.coords.validity()`.
If the work is not harmless under a null (see [Nulls](#nulls) below),
only do this when there are no nulls, and check `array.coords.is_valid(i)` otherwise.

For **Nested kinds**, this is using `array.coords`, with the range of each row (through `per_row`).
No need to null check here because of geoarrow validity spec.

```rust
#[inline(always)]
fn count_vertices<const INCLUDE_EMPTY: bool>(coords: &ValidCoords, vertices: Range<usize>) -> u32 { … }

pub fn point_coordinate_counts(array: &PointArray, …) -> PrimitiveArray<u32> {
    let coords = array.coords.ignoring_validity();
    let counts = (0..coords.len())
        .map(|i| count_vertices::<INCLUDE_EMPTY>(coords, i..i + 1))
        .collect();
    PrimitiveArray::from_vec(counts).with_validity(array.coords.validity().cloned())
}

pub fn line_string_coordinate_counts(array: &Untrimmed<LineStringArray>, …) -> PrimitiveArray<u32> {
    per_row_count(array.rows(), |i| {
        count_vertices::<INCLUDE_EMPTY>(array.coords(), array.rows().range(i))
    })
}
```

Do not turn points into one-coordinate lists to reuse the loop of the nested kinds.
That costs an offsets buffer for nothing,
and moves the nulls of the points to a level where they do not live.

## Vectorising simple kernels

A simple kernel on flat data (one result per point or per coordinate) can often compile to SIMD,
but a branch in the loop stops that from happening.
Strive for such kernels to have no branches at all.
Instead, compute a result for every row, including the nulls and empties,
and mask those results afterwards.

### Nulls

Compute over `coords.ignoring_validity()`, and give the output the validity of the input
(`combine_validities_and` for several inputs).
See `points_with_clamped_coords` in `native/fix_out_of_bounds.rs`.
This is only allowed when the computation is harmless on any value
(see "Values under a null" in the [Kernel implementation guide](kernel-implementation-guide.md#nulls)).
If it is not, only take the branchless route when there are no nulls,
with an `ALL_VALID` const generic
(see `pairwise` in `native/distance.rs`, where it is still called `NULLS`).

### Empties

NaN usually propagates by itself, so an empty point often gives the right result without any check.
If it does not, build a bitmap of the empties in a separate branchless pass,
and use it to fix up the output afterwards (e.g. by masking them out of the validity).
`bitmap_from_fn` in `native/coordinate_selection.rs` packs such a pass 64 bits at a time, so it compiles to SIMD too.

This only pays off if the rest of the loop is branch-free.
If the work per row branches anyway (e.g. looping over the parts of a nested geometry),
keep the check for nulls and empties in the loop instead.

## `early_coord_check`

A native kernel defaults to `early_coord_check="finite"`.
The check runs as a pass of its own, before the loop.
