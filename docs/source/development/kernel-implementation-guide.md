# Kernel implementation guide

This document is a guide for implementing compute kernels in `geopolars-compute`.
It holds what goes for every kernel.
Native kernels and GEOS kernels each have a guide of their own on top of this one:
the [Native kernel implementation guide](native-kernel-implementation-guide.md)
and the [GEOS kernel implementation guide](geos-kernel-implementation-guide.md).
Kernels forwarding to PROJ follow this guide alone.

## The target, and legacy names

These guides describe how a kernel *should* look.
Much of the code does not look like this yet,
so a pattern found in another kernel does not make it right.
(Where the guides and the code disagree, the guides win).

Some names in the code are older than the ones used here.
Read them as follows:

| In the code | In these guides |
| - | - |
| `unique_z`, `unique_m` | `include_z`, `include_m` |
| `NULLS` | `ALL_VALID`, which is its opposite |

The arguments are described in [Common arguments](../../../geopolars/docs/source/reference/common_arguments.rst).

## The entry point

A kernel has one public function for each geometry kind it accepts,
which takes the [view](view-types.md) of that kind:
`&PointArray`, `&LineStringArray`, `&Untrimmed<PolygonArray>`, and so on.
Avoid using a type-erased `Box<dyn Array>`! We use these nice types to show intent.
`geopolars-core` picks which of these to call (see [Dispatch](#dispatch)).

- A kernel that gives geometries returns a view too (e.g. a `PolygonArray`).
- Any other kernel returns a concrete Polars array (e.g. a `BooleanArray`).
- A kernel that can fail returns a `GeoComputeResult`.

This makes clear what data the kernel expects and gives,
which documents the code and leaves less room for bugs.

There are two exceptions, which take a `&GeometryArray` instead of one view per kind:

- A binary operation that allows comparing between different geometry types.
  One function per pair of kinds would be far too many.
- A GEOS kernel whose work does not depend on the kind,
  as converting to GEOS dispatches on the kind already (`ToGeosRows for GeometryArray`).

### Work on the arrays, not on single items

Prefer working on the arrays of a view directly,
rather than building something per item that copies or rebuilds offsets.
For nested types, saving the offsets and nestings of each item is all extra overhead.
Borrowing a single geometry with `array.get(i)` (e.g. a `ScalarPolygon`)
is fine if it improves legibility, as it copies nothing.
Not only does this keep the code more consistent,
it also means we only need to define helpers that work on the array side,
rather than needing duplications for them to also work on scalar values.

## Dispatch

Arguments reach the loop in three steps.

1. **The plugin** parses each string argument into its enum, at plan time,
   so a wrong value fails before any data is read.
2. **`geopolars-core`** matches on the `Kind`, builds the view of that kind,
   and calls the kernel's function for it.
   It passes the arguments on as enums and `bool`s.
   Here branching should happen normally (not const-generic),
   as this is not the hot loop.
3. **The kernel's public function** turns the arguments the loop needs into const generics,
   with `with_const_flags!`,
   and calls a private generic function that holds the loop.

Declare an enum to dispatch on with `const_enum!` in `geopolars-compute`,
as its dispatch only works within the crate that declares it.
Please try to use this pattern consistently, rather than reinventing it locally.

### Const or runtime branching

Branch prediction can be pretty good, but it is even nicer if the loop does not have to branch at all!
Whether an argument becomes a const generic depends on where it is used.
This is to keep the happy path quick, but not blow up all the baking combinatorics,
as each const flag doubles the copies of the loop.

The following should be dispatched as a const generic, so that the loop does not branch on it:

- It is **used for every item in the loop** independent of the row,
  (e.g. "do we have to check for NaNs?").
  The checks themselves, such as `early_coord_check` and `null_parts`, belong here too.

The following should be dispatched with a branch in the loop with a plan `if` or `match`,
as they are either edge cases, are not on the 'happy path',
or do not run for every (or most) rows:

- The path taken is **dependent on the content of the row**
- It is **used only outside the loop**.
  Making it const would only make more copies of the same function.
- **Used only on the rare path**, after hitting an error, an empty or an invalid state
  (`on_row_error`, `short_lines`, `empty_polygon_exterior`):
  a runtime check, on purpose.
  The happy path never reaches the branch,
  so a const generic would only blow up the binary size.

## Passes and allocation

These rules go for the kernel and for every helper it calls.

- **Size each output once**, up front (`Vec::with_capacity`, `SizeHint`).
- **Avoid allocating per row, part or ring.**
  A GEOS geometry is the exception (no way around that, sadly, as GEOS owns the data).
  One allocation per chunk, e.g. to build a trimmed view, is fine.
- **Do not collect a `Vec` only to iterate over it.** That's why we have iterators :)
- **Do not make a pass that only changes the type**,
  or copy a view only to reshape it.
  To leave out the z or m values (`include_z`, `include_m`),
  read the data as a type without them (before the hot loop).
  This is preferable over a const generic `bool`.
- **Reuse the levels that do not change.**
  A kernel that only changes the coordinates hands the input's offsets to `from_parts`
  (see `native/fix_out_of_bounds.rs`).
- **Copy contiguous ranges of coordinates** to pick or drop parts,
  and rebuild the offsets yourself.
  Do not use Polars' `take` or `gather` on a nested list,
  as this goes ring by ring, at about 1.5 µs per ring.

A helper that forces an allocation where a variant without one would do is a poor fit:
use or write the variant.

## Trimmed and untrimmed views

Views from `new` are trimmed, and a view a kernel builds has to be too
(see [Trimming](view-types.md#trimming)).

- A kernel that walks a whole level (e.g. every coordinate) takes a trimmed view,
  as an untrimmed one holds the items of rows outside the chunk too.
- A kernel that only reaches the levels through the offsets of its rows may take an `Untrimmed` view,
  where the trim is a noticeable part of its cost.
  It is the responsibility of the kernel to ensure that correctness is ensured.

## Backends

The backends (`native`, `geos` and `proj`) should be independent and not depend on each other.
An operation that needs more than one combines them
in a module next to the backends (e.g. `ogc_validity`) or in the plugin.

## Helpers

Before writing a helper, check that the views or `utils` do not have it already.
Where a helper goes depends on what it costs:

- `geopolars-views` holds only what costs nothing:
  O(1) per call, without allocating or writing
  (accessors, ranges, spans, O(1) checks such as `is_ring`).
  Building a view (constructors, trimming) belongs there too
  See [View Types](view-types.md#methods-on-the-views).
- `utils` in `geopolars-compute` holds every other helper that kernels share
  and that any backend could use, as it only needs the views,
  directly in that module, without sub-modules.
  This includes filters, and anything that loops over the coordinates, allocates or writes,
  such as `with_ranges_reversed`, `copy_ranges`, `gathered` and `with_null_at`.
  A helper that only one backend can use (e.g. one that needs a GEOS type)
  goes in the root module of that backend instead
  (`native/mod.rs`, `geos/mod.rs`, `rsgeo/mod.rs` or `proj.rs`).
- The shared infrastructure that kernels are built on is located in the root modules.
  A small helper in one of those modules that is not part of its infrastructure belongs in `utils`.
- A kernel module only holds helpers for that kernel alone.

The same logic in two modules is one helper,
in `utils`, or in the root module of the backend that alone can use it.
Functions that do similar things in a different way and differ on purpose must say why in a doc comment
(e.g. the empty point is x and y NaN for GEOS, but NaN on every axis for native).

A kernel module neither imports a helper from another kernel module nor exports one to it.
The exception is the kernel's 'main thing', what the kernel is essentially a loop around
(the function that computes the kernel's own result for one item (row, part or ring)),
such as `twice_signed_area` in `area`.

Keep a helper as private as you can.
Inline one that only forwards to another function, whose name says less than its body,
or that splits a loop so that its shape is harder to follow.
A helper that names a step of the kernel stays, even with a single caller.
Helpers get no tests of their own: the Python tests cover them through the expressions.

## Valid input

Kernels only take valid input, as view types.
These follow the GeoArrow spec, where only the outer level may have nulls.
No other level, including the coordinate values, can hold a null.
So do not check for nulls below the rows:
ensuring validity is the task of whoever builds the views.
This puts the burden on the kernel to give valid output too, though.

Constructors and sanitising functions are the exception:
the kernels that take `null_parts` or `check_null_coordinates`,
or that build geometries from parts or coordinates.
They take what may not be valid GeoArrow yet.

### NaN, infinity and empties

Validity does **not** shield you from NaN or infinite values, nor from *empty* geometries.
A valid polygon cannot have a null coordinate, but it can still have a NaN x.
These are all valid GeoArrow:

- Individual NaN or infinite values (e.g. `x = 32.1, y = NaN`).
- *Empty* points (NaN on every axis).
- Empty rings and parts (e.g. a polygon with an empty ring).

How to handle these is up to the kernel,
but the result for each has to match what the expression documents.
Rules and helpers for single coordinates (empty, not finite, outside the bounds)
are in `native/coordinate_selection.rs`.

## Nulls

Before doing any work, a kernel must first check for two special conditions for an all-null result:

- if there are no nulls (`null_count() == 0`), then it may take a faster path
  where it does not need to check for nulls internally (`ALL_VALID` const generic dispatch).
- if there are only nulls (`null_count == len()`), then it can simply directly return itself unchanged.

This is for most kernels, but there may of course be exceptions!

A resulting row can only be null when its input is null,
or when an error is handled as `on_row_error="null"`.
Never turn an error into a null or a default value another way
(`.ok()`, `unwrap_or_default()`, `Err(_) => None`). Use `on_row_error` instead.

- **Values under a null.**

Nulls should hold the same promise as those in Polars itself,
because a kernel may compute over the values under a null
and mask the result afterwards.
Nulls should therefore only cover up a harmless value:
no panics, no errors, and no indexing.

## Errors

We prefer being strict, and failing as early as possible.
Raise an error at plan time if you can
(e.g. `allow_geographic=False` on a geographic CRS).
If this is not possible, try to raise an error before the loop over the rows.
Only if the data really depends on the row data should you do error handling inside of the loop.

An error caused by a single row goes must be handled by `on_row_error`.
If it is set to `"raise"`, it should return it.
If it is `"null"`, it should null that row.
Regardless, it should never fail the whole chunk (e.g. with `?` inside `per_row`).

A kernel that aggregates (combines many rows into far fewer results)
asks the caller what to do with errors, with an `IGNORE_ERRORS` const `bool`.
The default is to return the error.

It should go without saying, but just in case:
Kernels should not panic.
Don't use any `unwrap`, `expect`,
and make sure you don't have any paths where you might index out of bounds,
overflow, divide by 0, or anything similar,
  unless validity guarantees it cannot fail.
  Then `unreachable!` and `debug_assert!` are fine.

## MaskedParts

The one exception to the validity rule is the view inside a `MaskedParts`.
The parts unset in its mask may hold arbitrary values where the input had a null.
Only code that takes a `MaskedParts` reads that view.
It skips those parts, or masks out what it computed for them.
It may hand the view to a plain kernel only if that kernel gives one value per part,
from that part alone, as the mask then drops the values of the unset parts.
Every kernel in turn has to give valid output.

## Documenting const flags

Describe each flag once, in a ``## `FLAG` `` sub-section under `# Const generic flags` in the module doc
(see `native/count_coordinates.rs`).
A runtime `bool` that dispatches to it has the same name in lowercase, so it needs no doc of its own.
