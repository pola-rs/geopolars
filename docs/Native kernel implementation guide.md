# Native kernel implementation guide

This document extends the [[Kernel implementation guide]].

## Keep branching out of the hot path

Try to keep branching outside of the core loop of a kernel.
Branch prediction can be pretty good, but it's even nicer if we don't have to!
Stuff like 'do we have to check for NaNs?' will have the same answer for every item,
so we can write that as a generic function instead, and dispatch to its specific versions.
The `with_const_flags!` macro is very useful here.
Be careful you don't blow it up with combinatorics though.

Describe each flag once, in a ``## `FLAG` `` sub-section under `# Const generic flags` in the module doc (see `native/count_coordinates.rs`).
A runtime `bool` that dispatches to it has the same name in lowercase, so it needs no doc of its own.

### Branching based on arguments

Most kernels take some arguments that determine how something should be done,
or if an optional step should be included.
Ideally, these would *also* be kept outside of the loop.
Some cannot, as they depend on the content of the row.
In such a case, you have no choice but to do the branching inside of the loop.

Cross-reference this with [the common arguments doc]("../docs/Common arguments.md").

## `compare_z` and `compare_m`

If the z and/or m values are dropped, you can simply re-interpret the data as a type without those dimensions.
This is preferable over dispatching this as a const generic bool,
though in some cases that is a valid approach too.
Whatever you do though, you should not do a first pass to simply change the type.

## Branching on errors, empties, or invalid states

Some argument say what to do on hitting an empty geometry, running into an error,
or running into an invalid state.
These are the exception, not the rule. Hence, they can take the penalty of runtime checking
(rather than blowing up the binary size).
These should *not* be dispatched using const generics.

Examples are `on_row_error`, `short_lines`, `empty_polygon_exterior`.

Note that this does not include functions on *checking for errors* in the first place, such as `early_coord_check`.
`null_parts` should also be dispatched statically.
If those are gated behind an argument, then that should still be dispatched in a const generic way.

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
The values under a null are arbitrary,
so this only works if the computation is harmless on any value!
It should not panic, no errors, no indexing with it.
If it is not, only take the branchless route when there are no nulls.
You can do this with a const generic `NO_NULLS: bool`
(see `pairwise` in `native/distance.rs`).

### Empties

NaN usually propagates by itself, so an empty point often gives the right result without any check.
If it does not, build a bitmap of the empties in a separate branchless pass,
and use it to fix up the output afterwards (e.g. by masking them out of the validity).
`bitmap_from_fn` in `native/coordinate_selection.rs` packs such a pass 64 bits at a time, so it compiles to SIMD too.

This only pays off if the rest of the loop is branch-free.
If the work per row branches anyway (e.g. looping over the parts of a nested geometry),
keep the check for nulls and empties in the loop instead.
