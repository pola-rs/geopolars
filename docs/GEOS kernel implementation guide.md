# GEOS kernel implementation guide

All kernels should have this shape:

```text
geoarrow -> GeosArray -> (geos kernel) -> GeosArray -> geoarrow

GeosArray = {Vec<GeosGeom>, validity bitmap }
GeosGeom = enum {geos geometry, empty, null, left_unchanged, right_unchanged}
```

## Use GeosRow types for trivial solutions

Converting a geometry to GEOS and calling GEOS on it is expensive.
Avoid creating GEOS geometries if the solution is trivial.
It is preferable to use a `GeosRow::Empty` or `GeosRow::Null` as its creation is much cheaper,
and it allows us to switch what we do without needing to inspect the GEOS data.

For example:
a GEOS kernel that can tell the result of a row without GEOS therefore gives it before converting,
such as a null or empty geometry on either side of a binary operation,
should short-circuit to give a `GeosRow::Empty` directly, bypass the conversion to GEOS.

## `reorient_rings` and `reverse_hole_direction`

GEOS operations that build new rings always give clockwise shells and counter-clockwise holes,
whatever the input's orientation. This held for (among others):

- intersection, union, difference and unary_union
- buffer, including buffer(0) and Point.buffer
- normalize
- make_valid on invalid input (the bowtie came back as two polygons, both with clockwise shells)

In these cases, `reorient_rings="ccw"` and `reverse_hole_direction=True` is a no-op, and can be skipped.

Note that this is not always the case though!
Operations that leave the rings alone keep the input's orientation:

- simplify returned a counter-clockwise shell with a counter-clockwise hole unchanged.
- make_valid on input that's already valid returns it as is, even when the hole has the wrong orientation.
  A valid polygon doesn't need any particular orientation, so this case passes straight through.

Test it out first before assuming anything.
