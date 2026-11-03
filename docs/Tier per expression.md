# Tier per operation

Which tier each operation uses, per geometry type.
`A` is a native kernel, `B` forwarded to an external library, `C` Polars expressions,
and `-` means the operation refuses that geometry.
If a constant value is returned for an operation, the value is also provided
(e.g. when passing a flag to get 0.0 for area of points, instead of erroring).
All dimensions of a geometry share a column.
Geometries usually carry a CRS, so the tier given is for a geometry with a CRS;
the comment says when one without a CRS takes a different tier.

| Operation | Point | | LineString | | Polygon | | MultiPoint | | MultiLineString | | MultiPolygon | |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| `area` | - | ANP | - | ANP | A/B | fwd `*_planar` `*_geodesic` | - | ANP | - | ANP | A/B | sum of parts |
| `area_planar` | - | ANP | - | ANP | A | shoelace; CRS ignored | - | ANP | - | ANP | A | sum of parts |
| `area_geodesic` | - | ANP | - | ANP | B | PROJ + GGL; needs a CRS | - | ANP | - | ANP | B | sum of parts |
| `length` | - | ANL | A/B | Reuses bits of `distance` [^2] | - | ANL | - | ANL | A/B | sum of parts | - | ANL |
| `centroid` | C | returns the point | A | length-weighted; keeps z/m | A | shoelace; z/m area-weighted | C | fwd `mean_coordinate` | A | length-weighted parts | A | area-weighted parts |
| `exterior` | - | | - | | A | ring 0 | - | | - | | A | LPP |
| `interior` | - | | - | | A | rings after the first | - | | - | | A | LPP |
| `add_holes` | - | | - | | A | holes: linestring(s) or multilinestring | - | | - | | - | |
| `boundary` | - | | C | | C | every ring | - | | C | LPP | C | LPP |
| `parts` | - | | - | | - | | C | [^4] | C | [^4] | C | [^4] |
| `join_line_strings` | - | | - | | - | | - | | A | [^5] | - | |
| `convex_hull` | - | | B | GEOS [^3], straight=null | B | GEOS [^3] [^1] | B | GEOS [^3] [^1] | B | GEOS [^3] [^1] | B | GEOS[^3] [^1] |
| `buffer` | B | GEOS; z and m dropped [^1] | B | | B | | B | | B | | B | |
| `offset_curve` | - | | B | GEOS; z and m dropped [^1] | - | | - | | B | | - | |
| `reverse` | - | | A | | A | `reverse_holes` | A | ignores `nested` | A | `nested` | A | `nested`, `reverse_holes` |
| `simplify` | C | returns the point | B | GEOS; z kept, m dropped | B | split by `buffer(0)` is a per-row error | C | returns the multipoint | B | | B | |
| `is_ring` | - | | A | ≥ 4 vertices, ends = x/y/z | - | | - | | A | parts chained into one ring | - | |
| `is_simple` | A | always `True` | B | GEOS; x/y only | B | every ring as a line; GEOS | A | no two points equal in x/y | B | GEOS, as a whole | B | every ring as a line; GEOS |
| `rings_overlap` | - | | B | hacked as polygon; GEOS | - | | - | | - | | - | |
| `is_ccw` | - | | A | CCW | A | | - | | A | every part combined | A | every part combined |
| `to_ccw` | - | | A | | A | `holes_cw` | - | | A | every part | A | every part |
| `to_cw` | - | | C | `to_ccw` + `reverse` | C | | - | | C | | C | |
| `close_rings` | - | | A | repeats the first vertex, or `snap_z`/`snap_m` | A | every ring | - | | A | every part | A | every ring |
| `is_close` | A | pairwise or against one | A | same coordinates in order | A | same rings | A | | A | same parts | A | same polygons and rings |
| `is_valid_geoarrow` | A | bounds from PROJ (plan-time) | A | | A | | A | EP | A | EP | A | EP |
| `null_out_invalid_geoarrow` | A | nulls what `is_valid_geoarrow` rejects | A | | A | | A | | A | | A | |
| `is_valid_ogc` | A | bounds from PROJ (plan-time) | A | | A/B | topology through GEOS | A | EP | A | EP | A/B | EP and GEOS |
| `null_out_invalid_ogc` | A | nulls what `is_valid_ogc` rejects | A | | A/B | | A | | A | | A/B | |
| `has_repeats_in_succession` | A | always `False` | A | | A | | A | | A | EP | A | EP |
| `remove_repeats_in_succession` | A | returns the input | A | | A | | A | | A | | A | |
| `filter_coordinates` | A | | A | | A | | A | | A | | A | |
| `fix_out_of_bounds` | A | bounds from PROJ (plan-time) | A | | A | | A | | A | | A | |
| `distance` | A/B | [^2] | - | | - | | - | | - | | - | |
| `mean_coordinate` | C | returns the point | C | | C | SCC | C | | C | | C | SCC |
| `translate` | A | impl differs for const or col | A | | A | | A | | A | | A | |
| `rotate` | A | | A | | A | | A | | A | | A | |
| `rotate_about_centroid` | - | | C | CAA | C | | - | | - | | - | |
| `skew` | A | `origin` may be a point column | A | | A | | A | | A | | A | |
| `skew_about_centroid` | - | | C | CAA | C | | - | | - | | - | |
| `scale` | A | impl differs for const or col | A | | A | | A | | A | | A | |
| `scale_about_centroid` | - | | C | CAA | C | | - | | - | | - | |
| `affine_transform` | A | `origin` may be a point column | A | | A | | A | | A | | A | |
| `snap_to_grid` | A | x/y/z only, m kept | A | | A | | A | | A | | A | |
| `bounds` | A | repeats its coordinate | A | | A | | A | LPP of `box` | A | LPP of `box` | A | LPP of `box` |
| `total_bounds` | A | one box for the column | A | | A | | A | | A | | A | |
| `set_crs` | C | [^4] | C | | C | | C | | C | | C | |
| `to_crs` | B | PROJ | B | PROJ | B | PROJ | B | PROJ | B | PROJ | B | PROJ |
| `is_geographic` | C | scalar, calls PROJ at plan | C | | C | | C | | C | | C | |
| `x` | C | `f64`, drops the CRS | C | `list[f64]` | C | `list²[f64]` | C | `list[f64]` | C | `list²[f64]` | C | `list³[f64]` |
| `y` | C | `f64`, drops the CRS | C | `list[f64]` | C | `list²[f64]` | C | `list[f64]` | C | `list²[f64]` | C | `list³[f64]` |
| `z` | C | like `x`; only with a `z` | C | | C | | C | | C | | C | |
| `m` | C | like `x`; only with an `m` | C | | C | | C | | C | | C | |
| `force_2d` | C | cast | C | | C | | C | | C | | C | |
| `force_3d` | C | `0.0`, NaN if empty | C | `list.eval` per nesting level | C | | C | | C | | C | |
| `without_m` | C | cast | C | | C | | C | | C | | C | |
| `force_m` | C | `0.0`, NaN if empty | C | `list.eval` per nesting level | C | | C | | C | | C | |
| `count_coordinates` | A | `0` if empty, unless `include_empty` | A | | A | SCC | A | | A | | A | SCC |
| `is_empty` | A | NaN `x` and `y` | A | no verts, or only empty if `check_nested` | A | | A | | A | | A | |
| `wrap_longitude` | C | needs geographic CRS | C | | C | | C | | C | | C | |
| `to_wkb` | A | `wkb` crate | A | | A | | A | | A | | A | |
| `from_wkb` | A | `wkb` crate | A | | A | | A | promotes a point | A | promotes a linestring | A | promotes a poly |
| `to_wkt` | A | `wkt` crate | A | | A | | A | | A | | A | |
| `from_wkt` | A | `wkt` crate | A | | A | | A | promotes a point | A | promotes a linestring | A | promotes a poly |

In this table:

- GGL = 'geographiclib'
- CRS = Coordinate Reference System, stored in metadata
- EP = true when every part passes the single variant's check, one `bool` per row
  (e.g. a multipolygon is only valid if each of its polygons is)
- ANP = refused, unless `allow_non_polygons` is set, then C returning `0.0` (null stays null)
- ANL = refused, unless `allow_non_lines` is set, then C returning `0.0` (null stays null)
- LPP = returns a list, one result per part (computed as for the single variant)
- CAA = composed of `centroid` + `affine_about`
- SCC = skips the closing coordinate of each ring

[^1]: GEOS does not support m values (for this operation),
      so its value is dropped, and an m-less variant is returned.
[^2]: user chooses method, all of which differ in implementation method.
      Planar: A; geodesic: PROJ + GGL, needs a CRS.
[^3]: Convex hull only cares around the points, not the actual geometry.
      See the implementation in `geopolars-compute`.
[^4]: Metadata relabeling
[^5]: Also takes a `list[LineString]`.
