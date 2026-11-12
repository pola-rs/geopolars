# Move dispatching from `geopolars` to `geopolars-core`

Every plugin expression in `geopolars/src/expr/` still dispatches on `Kind` itself.
The views moved to `geopolars-views`, so that `geopolars-core` can call the kernels of `geopolars-compute`.

## Shared dispatch helpers

Moving these first means the expressions that use them only need their imports changed.

- [x] `geoarrow::geometry_array` → `GeometryArray::new` (`Kind` → `GeometryArray`): used by `buffer`, `relate`, `overlay`, `is_topologically_equal`, `shortest_line_between`, `hausdorff_line_between`
- [x] `geoarrow::for_each_geometry` → `GeometryArray::for_each_geometry` (per-row `Geometry`): used by `to_wkb`, `to_wkt`
- [x] The `geoarrow::decode` builder → `GeometryBuilder` (matches on the target `Kind`): used by `from_wkb`, `from_wkt`

## One geometry column, matching on `Kind` in the expression itself

- [x] `add_holes`: `add_holes`
- [x] `add_parts`: `add_parts`
- [x] `affine`: `affine`, `affine_about`, `affine_then_shift`
- [x] `area`: `area_planar`, `area_geodesic`
- [x] `bounds`: `bounds`, `total_bounds`
- [x] `centroid`: `centroid`
- [x] `close_rings`: `close_rings`
- [x] `convex_hull`: `convex_hull`
- [x] `count_coordinates`: `count_coordinates`
- [x] `crs`: `to_crs`
- [x] `filter_coordinates`: `filter_coordinates`
- [x] `fix_out_of_bounds`: `fix_out_of_bounds`
- [x] `has_repeats_in_succession`: `has_repeats_in_succession`
- [x] `is_ccw`: `is_ccw`
- [x] `is_empty`: `is_empty`
- [x] `is_ring`: `is_ring`
- [x] `is_simple`: `is_simple`
- [x] `is_valid`: `is_valid_geoarrow`, `is_valid_ogc`, `null_out_invalid_geoarrow`, `null_out_invalid_ogc`
- [x] `join_line_strings`: `join_line_strings`
- [x] `length`: `length`
- [x] `line_interpolate_point`: `line_interpolate_point`
- [x] `offset_curve`: `offset_curve`
- [x] `remove_repeats_in_succession`: `remove_repeats_in_succession`
- [x] `reverse`: `reverse`
- [ ] `rings`: `exterior`, `interior`
- [ ] `simplify`: `simplify`
- [ ] `snap_to_grid`: `snap_to_grid`
- [ ] `sort_parts`: `sort_parts`
- [ ] `to_ccw`: `to_ccw`
- [ ] `buffer`: `buffer` (only picks the multi or single output; the array goes through `geometry_array`)

## Two geometry columns

- [ ] `distance`: `distance` (`aligned_pairs` + `Kind` match)
- [ ] `is_equal`: `is_equal` (`aligned_pairs` + `Kind` match)
- [ ] `rings_overlap`: `rings_overlap` (`aligned_pairs` + `Kind` match)
- [ ] `relate`: `relate` (via `geometry_array`)
- [ ] `overlay`: `intersection`, `difference` (via `geometry_array`, plus a `Kind` match on the output)
- [ ] `is_topologically_equal`: `is_topologically_equal` (via `geometry_array`)
- [ ] `shortest_line_between`: `shortest_line_between` (via `geometry_array`)
- [ ] `hausdorff_line_between`: `hausdorff_line_between` (via `geometry_array`)

## Encoding

- [ ] `wkb`: `to_wkb` (via `for_each_geometry`), `from_wkb` (via `decode`)
- [ ] `wkt`: `to_wkt` (via `for_each_geometry`), `from_wkt` (via `decode`)

## Constructors (dispatch on the output kind, not an input geometry)

- [ ] `construct`: `linestring`, `polygon`, `multipoint`, `multilinestring`, `multipolygon`, `linestring_coords`, `polygon_coords`, `polygon_nested_coords`, `multipoint_coords`, `multilinestring_coords`, `multipolygon_coords`, `multipolygon_nested_coords`, `polygon_rings_in_separate_columns`

## Not on the list

`aligned_pairs` and `single_chunk_masked_pairs` in `geoarrow/pairs.rs` only line up chunks and never look at the `Kind`.
Whether they move to core is a separate decision.
