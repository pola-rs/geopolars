
# Roadmap

## Core IO

- [x] WKB encoding and decoding
- [x] WKT encoding and decoding
- [x] GeoParquet read
- [x] GeoParquet write
- [x] GeoParquet bbox covering (written on request, read as a box)

Later goals:

- [ ] GeoJson read
- [ ] PostGIS read
- [ ] GDAL vector files
- [ ] Interop with arrow

## Per-geometry properties and ops

- [x] Area
- [x] Length
- [x] getters for coordinates (z/m)
- [x] CRS: opaque SRIDs refused for reprojection
- [x] is_empty for all geo
- [x] validity checking functions
- [x] Handling closing of loop consistently for polygon
- [x] mean coordinate
- [x] centroid
- [x] convex hull
- [x] simplify/decimate: `simplify`

affine transforms:

- [x] translate
- [x] rotate
- [x] scale
- [x] skew
- [x] transform (with a matrix): `affine_transform`
- [x] shift (by a value in a different column)
- [x] rotate around coordinate from another column
- [x] scale around coordinate from another column
- [x] skew around a coordinate from another column
- [x] rotate around own centroid (can use the above)
- [x] scale around own centroid (can use the above)
- [x] skew around own centroid (can use the above)

transfer transforms

- [x] line_interpolate_point
- [ ] project

geometry editing

- [x] polygon construction from linestrings
- [x] reverse lines
- [x] make lines cw/ccw: `to_cw`, `to_ccw`
- [ ] drop_dims(z: bool, m: bool)
- [x] to_linestring() takes list of points, multipoint, linestring: `line_string_from_vertices`
- [x] split a multi-geometry into a list of its parts: `parts`
- [x] sort the parts of a geometry: `sort_parts`
- [ ] is_continuous (for multilinestring)

repair:

- [ ] repair: drop overlaps, with the options for reltol etc.
- [x] Close rings expr, can use this inside of 'repair'

## Two-column operations

- [x] Adding additional holes to polygons from linestrings: `add_holes`
- [x] Adding list[linestring] to multilinestring/list[polygon] to multipolygon etc.: `add_parts`
- [x] equality: (are two shapes the same?): `is_equal`, `is_topologically_equal`
- [x] shortest line between two geometries: `shortest_line_between`
- [x] line spanning the Hausdorff distance between two geometries: `hausdorff_line_between`
- [ ] Adding list[linestring] to multilinestring/list[polygon] to multipolygon etc.

Classic binary predicates:

- [x] DE-9IM matrix: `relate`
- [x] intersect: `intersects`
- [ ] contains
- [ ] within
- [ ] covers
- [ ] covered_by
- [ ] touches
- [ ] crosses
- [ ] overlaps
- [ ] disjoint
- [ ] distance_to

creating new geometries

- [x] intersection: `intersection`
- [ ] union
- [ ] difference

## Spatial hashing

- [ ] Make a fast spatial hash that works with arrow directly
- [ ] expression that drops all the repeated points in a geometry (out of order)
- [ ] Expression to get neighbours

## Architecture

- [ ] do the matching on kind in geopolars_core, not in the plugin.
- [ ] Ask Claude to check how our native kernels compare to GEOS' source. Is there any numeric-specific (e.g dealing with subnormal numbers) that they do, and we don't?
- [ ] Ask claude to make a matrix for what we cover compared to geopandas/GEOS for internal referencing
- [ ] The doc decorator does not work with popup hints. That's a dealbreaker.
- [ ] Instead of doc decorator, see if we can make a central page that we reference to (e.g 'how do we handle errors in geos')
- [ ] Make all GEOS kernels the shape as described in the compute readme (Always store of geos things of things)
