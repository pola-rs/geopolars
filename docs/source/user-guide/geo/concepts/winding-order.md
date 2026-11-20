# Winding order

Polygons in GeoArrow are represented as a series of rings.
One forming the exterior hull, and all others as holes cut away from that hull.
These rings run either *counter-clockwise* or *clockwise*.
This is called their winding order, or orientation.

The common convention, used by OGC and GeoParquet,
is a counter-clockwise exterior with clockwise holes.

## GeoPolars takes any winding order

GeoPolars does not require any winding order.
It accepts rings that run either way,
and operations such as `area` and `intersection` do not depend on it.
For example, a hole is always subtracted from the area, whichever way it runs:

{{code_block('user-guide/geo/concepts/winding-order','df',['DataFrame','geo.from_wkt','PolygonXY'])}}

{{code_block('user-guide/geo/concepts/winding-order','is_ccw',['geo.is_ccw','geo.area'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/winding-order"
--8<-- "python/user-guide/geo/concepts/winding-order.py:df"
--8<-- "python/user-guide/geo/concepts/winding-order.py:is_ccw"
```

GeoPolars also keeps rings in the order they come in, wherever it can,
as reversing them means copying more data.

GeoPolars leverages GEOS for many operations, which does have a winding order.
The rings GEOS builds usually run the other way from the OGC convention,
but not always (see [New rings](#new-rings)).

If winding order is important to you,
you can always inspect the current winding order of a geometry row,
and manually re-orient the winding orders.
You can also supply arguments to operations with GEOS that automatically do this upon converting back
from the GEOS representation to the GeoArrow representation.

## Checking and changing the winding order

Some tools do need a particular winding order.
For example, GeoJSON (RFC 7946) asks for counter-clockwise exteriors,
and so does GeoParquet when its edges are spherical.

- `is_ccw` checks whether a geometry follows the common convention.
- `to_ccw` makes it follow the convention, reversing only the rings that run the wrong way.
- `to_cw` does the opposite: a clockwise exterior, with counter-clockwise holes.

{{code_block('user-guide/geo/concepts/winding-order','to_ccw',['geo.to_ccw','geo.is_ccw'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/winding-order"
--8<-- "python/user-guide/geo/concepts/winding-order.py:to_ccw"
```

They also take linestrings, which they treat as closed.
With `holes_cw=False` (or `holes_ccw=False` for `to_cw`), the holes run the same way as the exterior.

## New rings

Some operations make rings of their own, such as `buffer`, `intersection`, `difference` and `convex_hull`.
These run in whatever order GEOS gives them,
which is usually a *clockwise* exterior with counter-clockwise holes.

Pass `new_ring_orientation="ccw"` (or `"cw"`) to get the exterior in that order instead.
This costs very little:
GeoPolars knows which way GEOS gives most rings, so it can reverse them without checking.
The holes then run the other way, unless you pass `new_holes="same"`.

{{code_block('user-guide/geo/concepts/winding-order','new_rings',['geo.buffer','geo.is_ccw'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/winding-order"
--8<-- "python/user-guide/geo/concepts/winding-order.py:new_rings"
```
