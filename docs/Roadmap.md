
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
- [ ] Handling closing of loop consistently for polygon
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

- [ ] interpolate
- [ ] project

geometry editing

- [ ] Linestring joining
- [x] polygon construction from linestrings
- [ ] Adding additional holes to polygons from linestrings
- [x] reverse lines
- [x] make lines cw/ccw: `to_cw`, `to_ccw`
- [ ] drop_dims(z: bool, m: bool)
- [x] to_linestring() takes list of points, multipoint, linestring: `line_string_from_vertices`
- [ ] is_continuous (for multilinestring)

repair:

- [ ] repair: drop overlaps, with the options for reltol etc.
- [x] Close rings expr, can use this inside of 'repair'
- [ ] expression that drops all the repeated points in a geometry (out of order)

other

- [ ] expr that joins `list[linestring]` or multilinestring into one long string.
      Bool flag on requiring ends to be coinciding (if true, must return multilinestring,
      otherwise linestring).
      Bool flag on deleting repeated overlapping point

## Binary predicates

- [ ] equality: (are two shapes the same?)
- [ ] intersect
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

- [ ] intersection
- [ ] union
- [ ] difference
