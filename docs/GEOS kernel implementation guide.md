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
