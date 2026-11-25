# Plan: drop the early coordinate check from planar `area`

Untracked working file. Do not commit it.

## Goal

Planar `area` no longer runs `early_coord_check`.
Geodesic `area` keeps it.
The argument stays on the Python API, and has no effect when measuring planar.

The docs were rewritten first, in commit e0db84e, and are the source of truth:

- `docs/source/development/geo/kernel-implementation-guide.md`, "NaN, infinity and empties".
- `docs/source/development/geo/native-kernel-implementation-guide.md`, "`early_coord_check`".
- `geopolars/docs/source/reference/common_arguments.rst`, `early_coord_check` and `on_out_of_bounds`.
- `docs/source/user-guide/geo/concepts/nulls-nan-and-empty.md`, "Non-finite".

Read those four sections before touching code. Where this plan and the docs disagree, the docs win.

## Why

The check is a pass over x and y that finds a NaN or infinite coordinate, so that the row gets NaN without being measured.
It exists to skip expensive work.
The planar shoelace formula reads the same x and y once and costs about as much as the check,
so on planar the check only adds a pass. The shoelace cannot turn a NaN or infinite coordinate into a finite number,
so a planar row with one comes out NaN or infinite by itself.
The geodesic ring sum (geographiclib) is expensive per vertex, so the check pays for itself there.

The bounds check needs no change. In `is_outside` in `geopolars-compute/src/native/coordinate_selection.rs`,
an infinite coordinate counts as outside and a NaN never does. Keep that.

## Step 1: small fixes in the docs just committed

- `docs/source/development/geo/kernel-implementation-guide.md`:
  the link `nulls-nan-and-empty.md#Non-Finite` must be `nulls-nan-and-empty.md#non-finite`
  (mkdocs lowercases heading anchors; compare the existing `#empty-geometries` link a few lines up).
  The sentence `See [the nulls and empty guide](...)` has no full stop.
- `docs/source/development/geo/native-kernel-implementation-guide.md`:
  "shoelace formale" must be "shoelace formula", and that sentence has no full stop.

Nothing else in the docs changes.

## Step 2: the kernel, `geopolars-compute/src/native/area.rs`

Add an associated const to the `AreaMetric` trait, after `type Chunk`:

```rust
/// Whether the kernel runs `early_coord_check` before measuring a row.
/// `false` where measuring costs no more than the check does.
const CHECKS_COORDS_EARLY: bool;
```

- `impl AreaMetric for Planar` in the same file: `const CHECKS_COORDS_EARLY: bool = false;`
- `impl AreaMetric for GeodesicMetric` in `geopolars-compute/src/metric.rs` (around line 81):
  `const CHECKS_COORDS_EARLY: bool = true;`

In `areas_with`, directly before the `with_const_flags!` call, add:

```rust
// A metric that does not check gets the `Off` path, whatever the user passed.
let early_coord_check = if M::CHECKS_COORDS_EARLY {
    early_coord_check
} else {
    EarlyCoordCheck::Off
};
```

Change nothing else in the function. With `EarlyCoordCheck::Off`,
`every_xy_passes_at` in `areas_with_failed_check` already returns `true` without touching the coordinates,
so the bounds check and `metric.check_row` run exactly as before.

Do not add a post-check on the result, do not change `twice_signed_area`,
and do not change `areas_with_failed_check` or the bounds code.
The module doc at the top of the file stays as it is.

## Step 3: the Python docstring, `geopolars/geopolars/geo/area.py`

In the `area` docstring, the paragraph after `{early_coord_check}` reads:

```text
A row that fails the check gets NaN, without being measured.
```

Replace it with:

```text
Only `"geodesic"` runs the check, as the planar formula costs no more than the check does.
A geodesic row that fails it gets NaN, without being measured.
Measured planar, a NaN or infinite coordinate gives NaN or infinity by itself.
```

Keep the argument, its default, and its `validate_one_of` line as they are.
The `{early_coord_check}` snippet in `geopolars/geopolars/_docs.py` stays as it is.

## Step 4: tests, `geopolars/tests/unit/operations/test_area.py`

`test_a_non_finite_x_gives_a_non_finite_area` (around line 357) currently expects NaN for the checked modes on both paths.
Planar now measures every row, so for planar every mode gives a non-finite number, not necessarily NaN.
Change the loop body to:

```python
        got = _areas(checked, df)
        assert got[0] == square
        if distance_method == "geodesic" and early_coord_check != "off":
            assert math.isnan(got[1])
        else:
            assert not math.isfinite(got[1])
```

and the docstring to:

```python
    """A checked geodesic row gets NaN without being measured.
    A planar row is always measured, and comes out non-finite."""
```

`test_a_nan_is_not_outside_the_bounds` (around line 474) keeps its assertions.
Change its docstring to `"""NaN passes the bounds, and the measurement gives NaN by itself."""`.

`test_the_early_check_keeps_an_infinite_y_from_proj` is geodesic and stays as it is.

Add no other tests.

## Step 5: checks

Run, in this order, and fix what fails:

```sh
make test-rust
UV_OFFLINE=1 make test
```

`make test-rust` is the only way to run the Rust tests here. Do not run plain `cargo test`.
`UV_OFFLINE=1 make test` rebuilds the plugin and runs the Python tests.

## Step 6: benchmark

This is a kernel change in a performance-sensitive codebase, so it needs numbers before it is done.
Use the `bench-compare` skill on this branch. Cases, each in memory and streaming:

- planar `area` on polygons with clean coordinates;
- planar `area` on polygons where about one row in ten holds a NaN vertex;
- geodesic `area` on clean polygons in EPSG:4326, as a control that should not move.

Prepare everything, then stop at the skill's ready banner and wait for the user to say yes before timing.
Every number must come from the release builds the skill makes.
Report the change per case. Expected: planar clean data faster, planar NaN data about neutral
(a NaN row now runs the shoelace instead of the check pass, which cost about the same),
geodesic unchanged. Flag any regression over about 5%.

## Step 7: commit

One commit, a single plain-text summary line such as `drop the early coordinate check from planar area`.
No markdown, no trailers, no attribution lines.

## Out of scope

- No change to the bounds check, to `every_xy_passes_at`, or to any other kernel.
- No change to `tier-per-expression.md`.
- No new tests beyond the edits in step 4.
