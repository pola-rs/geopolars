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

## `on_row_error`

This should be dispatched through a const generic, because a user would most likely use this to speed-up.

> [!NOTE] Legacy
> this was previously named `ignore_per_row_errors`. These should be migrated over to the new name.

## `compare_z` and `compare_m`

If the z and/or m values are dropped, you can simply re-interpret the data as a type without those dimensions.
This is preferable over dispatching this as a const generic bool,
though in some cases that is a valid approach too.
Whatever you do though, you should not do a first pass to simply change the type.
