# Nulls, NaN and empty geometries

There are three ways for a geometry to be *missing* something,
and GeoPolars treats each of them differently:

- A **null** can be used to represent a *missing* or *invalid* geometry.
  This should ring a bell for everyone familiar with Polars :)
- An **empty** geometry is a geometry without any coordinates, such as `POLYGON EMPTY`.
  Note that nested geometries like polygons and multis can be non-empty at the highest level,
  but contain empty geometries at their nested levels.
  As such, a multipolygon that contains only empty polygons is technically not empty itself!
- A **NaN** coordinate is a vertex whose x or y is not a number.
  NaNs are really only for numeric values.
  This does not stop a geometry from having vertices in it with NaNs though!
  A polygon where one of the vertices contains a NaN is perfectly valid (though often times undesirable!)

Below, we compare a square, an empty polygon and a null:

{{code_block('user-guide/geo/concepts/nulls-nan-and-empty','df',['geo.is_empty','geo.area','geo.centroid'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/nulls-nan-and-empty"
--8<-- "python/user-guide/geo/concepts/nulls-nan-and-empty.py:df"
```

See if this makes sense in your mental model:
The empty polygon is still a polygon. It has an area, which is 0.
Its centroid would also be the empty point.
If it does, then you got it!

## Nulls

A null geometry works like any other null in Polars.
You can find nulls with `is_null`, and remove them with `drop_nulls` or `filter`.
An operation on a null gives a null, and so does a check:
`is_empty` on a null is neither `True` nor `False`, but null.

GeoArrow only allows *nulling on the outer nesting*.
This means that either a *whole* geometry is null, or nothing inside of it is null.
A vertex, a ring or a single coordinate inside a geometry cannot be null
(see [validity](validity.md)).
So when you build a geometry from parts and one of them is null,
the whole row becomes null:

{{code_block('user-guide/geo/concepts/nulls-nan-and-empty','null_inside',['geo.point'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/nulls-nan-and-empty"
--8<-- "python/user-guide/geo/concepts/nulls-nan-and-empty.py:null_inside"
```

`add_parts`, `add_holes` and `polygons_from_rings_in_separate_columns` can leave the null part out instead,
with `null_parts="drop"`.

An operation can also *give* a null, when it fails on a row while `on_row_error="null"` is set,
as you will see [below](#non-finite-and-inner-empties-in-geos-backed-kernels).

## Empty geometries

An empty geometry has no coordinates.
It is [valid](validity.md), both by GeoArrow and by OGC, so operations accept it.
You find empty geometries with `is_empty`.

Nested kinds are stored as an empty list,
and the **empty point** is stored as a point with a NaN on every axis it has.
A point with NaN on only some axes, such as `(NaN, 3)`, is not empty.
It is simply a point that is not finite!

{{code_block('user-guide/geo/concepts/nulls-nan-and-empty','empty_point',['geo.is_empty','geo.is_valid_geoarrow'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/nulls-nan-and-empty"
--8<-- "python/user-guide/geo/concepts/nulls-nan-and-empty.py:empty_point"
```

An empty point can also show up *inside* another geometry.
In a multipoint, this is always legal, but it is not always valid according to OGC.
See the doc on OGC validity for more details.

By default, `is_empty` only looks at the outside:
a linestring with only empty points still has vertices, so it is not empty.
With `check_nested=True`, `is_empty` also looks inside,
and counts it as empty.
This is slower, as it has to read every coordinate.
Below, the `gap` route has one empty point in the middle,
and the `only gaps` route has nothing but empty points:

{{code_block('user-guide/geo/concepts/nulls-nan-and-empty','nested',['geo.line_string_from_columns','geo.is_empty','geo.length'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/nulls-nan-and-empty"
--8<-- "python/user-guide/geo/concepts/nulls-nan-and-empty.py:nested"
```

## Non-finite

A NaN or an infinity in a coordinate is viral and permeates into every result that depends on it.
For an operation like distance (and thus line length), the type follows naturally.
The distance to an infinitely far vertex is also infinite.
In other cases, such as an area, they can morph.
The area of a polygon with both a -inf and +inf can become NaN.
Some operations will raise errors (that can be handled by `on_row_error`) when they find a NaN.
This can be for whatever reason, such as an external tool (GEOS) not supporting it,
or because a precondition for a calculation is now not met.

No matter how a non-finite number is handled though,
it is guaranteed that a NaN never comes out as a plausible finite number.
You will always be able to find the invalid data, and it is never fixed silently.

!!! warning

    This holds with the default `early_coord_check`.
    With `early_coord_check="off"`, a GEOS-backed operation cannot guarantee that the result stays non-finite,
    as GEOS sometimes handles these internally inconsistently.

Non-finite numbers can also have implications on the validity of a geometry.
It also becomes impossible to tell whether two shapes cross.
Such a geometry is therefore not valid by OGC.
`is_valid_geoarrow(..., check_finite_coordinates=True)` finds them in a single, cheap pass.

## Non-finite and inner empties in GEOS-backed kernels

Operations that use [GEOS](https://libgeos.org/), such as `buffer` and `simplify`,
cannot handle these coordinates at all,
nor can it handle empty points inside a geometry.
GeoPolars checks for both before a row goes to GEOS,
and raises an error if it finds one.
With `on_row_error="null"`, such a row becomes null instead:

{{code_block('user-guide/geo/concepts/nulls-nan-and-empty','geos',['geo.simplify'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/nulls-nan-and-empty"
--8<-- "python/user-guide/geo/concepts/nulls-nan-and-empty.py:geos"
```

The check costs an extra pass over the coordinates.
With `early_coord_check`, you can turn it off, or only check for NaN and infinity.
See *Common arguments* in the API reference for what GEOS then does with such a row.

To get rid of these coordinates instead, use `filter_coordinates`.
By default, it drops every coordinate that is NaN, infinite, or an empty point.
It can also replace them with the empty point, or make the row null.

{{code_block('user-guide/geo/concepts/nulls-nan-and-empty','filter',['geo.filter_coordinates'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/nulls-nan-and-empty"
--8<-- "python/user-guide/geo/concepts/nulls-nan-and-empty.py:filter"
```

Dropping vertices can leave a line too short to be a line, or a ring too short to be a ring.
The `short_lines` argument decides what happens then.
A line that loses every vertex becomes empty.

## Summary

| | null | empty | NaN or infinity |
| - | - | - | - |
| What it is | no geometry | a geometry without coordinates | a coordinate that is not a number |
| Find it with | `is_null` | `is_empty` | `is_valid_geoarrow(..., check_finite_coordinates=True)` |
| Valid by OGC | (null) | yes | no |
| Operations give | null | an empty geometry, 0, or null | NaN, or an error for GEOS operations |
| Get rid of it with | `drop_nulls` | `filter(~is_empty(...))` | `filter_coordinates` |
