//! Adding parts to geometries, from another column.
//! See `add_parts` in Python for the rules.

use geopolars_compute::inner_nulls::{
    drop_null_parts_of_multi_line_strings, drop_null_parts_of_multi_points,
    drop_null_parts_of_multi_polygons, propagate_inner_nulls_of_line_strings,
    propagate_inner_nulls_of_multi_line_strings, propagate_inner_nulls_of_multi_points,
    propagate_inner_nulls_of_multi_polygons, propagate_inner_nulls_of_points,
    propagate_inner_nulls_of_polygons,
};
use geopolars_compute::native::{
    line_strings_are_empty, multi_line_strings_with_added_parts,
    multi_line_strings_with_kept_parts, multi_points_with_added_parts,
    multi_points_with_kept_parts, multi_polygons_with_added_parts, multi_polygons_with_kept_parts,
    points_are_empty, polygons_are_empty,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, List, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, UncheckedStorage,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::construct::{
    ensure_fits, fields_of, one_line_per_row, one_point_per_row, one_polygon_per_row, paired_rows,
    storage_of,
};
use crate::geoarrow::{describe, Geo, Kind};

/// `null_behavior` of `add_parts`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum NullBehavior {
    Ignore,
    Propagate,
}

/// `empty_behavior` of `add_parts`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EmptyBehavior {
    Keep,
    Skip,
}

/// The keyword arguments of `add_parts`.
/// These have to be mirrored on the python side.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddPartsKwargs {
    null_behavior: NullBehavior,
    empty_behavior: EmptyBehavior,
}

/// The kind of the parts of a `kind`, and the multi-geometry of them.
fn part_kinds(kind: Kind) -> (Kind, Kind) {
    match kind {
        Kind::Point | Kind::MultiPoint => (Kind::Point, Kind::MultiPoint),
        Kind::LineString | Kind::MultiLineString => (Kind::LineString, Kind::MultiLineString),
        Kind::Polygon | Kind::MultiPolygon => (Kind::Polygon, Kind::MultiPolygon),
    }
}

/// The multi-geometries `add_parts` gives back,
/// and whether the parts are a list of them per row (a list or a multi-geometry),
/// rather than one part.
fn parts_added(input_fields: &[Field]) -> PolarsResult<(Geo, bool)> {
    let geo = describe(input_fields[0].dtype())?;
    let (part, multi) = part_kinds(geo.kind);
    let parts = input_fields[1].dtype();
    let found = match parts {
        DataType::List(inner) => describe(inner).ok().filter(|found| found.kind == part),
        dtype => describe(dtype)
            .ok()
            .filter(|found| found.kind == part || found.kind == multi),
    };
    let Some(found) = found else {
        polars_bail!(
            SchemaMismatch: "expected the parts as one `{}` per row, a list of them, or a `{}`, got: {}",
            part.display(), multi.display(), parts
        );
    };
    ensure_fits(
        ("geometry", geo.dim, geo.metadata),
        ("parts", found.dim, found.metadata),
    )?;
    let listed = found.kind == multi || matches!(parts, DataType::List(_));
    Ok((Geo::new(multi, geo.dim, geo.metadata.clone()), listed))
}

fn kernel_output_type_add_parts(input_fields: &[Field]) -> PolarsResult<Field> {
    let (geo, _) = parts_added(input_fields)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

/// The rows of the result: null where `base` is,
/// and with [NullBehavior::Propagate] also where `parts` is.
fn rows_validity(base: &List, parts: &List, null_behavior: NullBehavior) -> Option<Bitmap> {
    match null_behavior {
        NullBehavior::Ignore => base.validity().cloned(),
        NullBehavior::Propagate => combine_validities_and(base.validity(), parts.validity()),
    }
}

fn not_empty(empty: BooleanArray) -> Bitmap {
    !empty.values()
}

/// One chunk of the parts to add to points,
/// holding one point per row, or with `listed=true` a list of them.
fn point_parts(
    chunk: &dyn Array,
    listed: bool,
    kwargs: AddPartsKwargs,
) -> PolarsResult<MultiPointArray> {
    let storage = UncheckedStorage::new(chunk)?;
    let parts = match (listed, kwargs.null_behavior) {
        (false, _) => one_point_per_row(propagate_inner_nulls_of_points(storage)),
        (true, NullBehavior::Ignore) => drop_null_parts_of_multi_points(storage),
        (true, NullBehavior::Propagate) => propagate_inner_nulls_of_multi_points(storage),
    };
    Ok(match kwargs.empty_behavior {
        EmptyBehavior::Keep => parts,
        EmptyBehavior::Skip => {
            let points = PointArray::new(Coords::from_valid(parts.coords.clone(), None));
            let keep = not_empty(points_are_empty(&points));
            multi_points_with_kept_parts(parts, &keep)
        }
    })
}

/// [point_parts], for linestrings.
fn line_parts(
    chunk: &dyn Array,
    listed: bool,
    kwargs: AddPartsKwargs,
) -> PolarsResult<MultiLineStringArray> {
    let storage = UncheckedStorage::new(chunk)?;
    let parts = match (listed, kwargs.null_behavior) {
        (false, _) => one_line_per_row(propagate_inner_nulls_of_line_strings(storage)),
        (true, NullBehavior::Ignore) => drop_null_parts_of_multi_line_strings(storage),
        (true, NullBehavior::Propagate) => propagate_inner_nulls_of_multi_line_strings(storage),
    };
    Ok(match kwargs.empty_behavior {
        EmptyBehavior::Keep => parts,
        EmptyBehavior::Skip => {
            let lines = LineStringArray::from_parts(
                List::from_offsets(parts.lines.offsets().clone(), None),
                parts.coords.clone(),
            );
            let keep = not_empty(line_strings_are_empty::<false>(&lines.into()));
            multi_line_strings_with_kept_parts(parts, &keep)
        }
    })
}

/// [point_parts], for polygons.
fn polygon_parts(
    chunk: &dyn Array,
    listed: bool,
    kwargs: AddPartsKwargs,
) -> PolarsResult<MultiPolygonArray> {
    let storage = UncheckedStorage::new(chunk)?;
    let parts = match (listed, kwargs.null_behavior) {
        (false, _) => one_polygon_per_row(propagate_inner_nulls_of_polygons(storage)),
        (true, NullBehavior::Ignore) => drop_null_parts_of_multi_polygons(storage),
        (true, NullBehavior::Propagate) => propagate_inner_nulls_of_multi_polygons(storage),
    };
    Ok(match kwargs.empty_behavior {
        EmptyBehavior::Keep => parts,
        EmptyBehavior::Skip => {
            let polygons = PolygonArray::from_parts(
                List::from_offsets(parts.polygons.offsets().clone(), None),
                parts.rings.clone(),
                parts.coords.clone(),
            );
            let keep = not_empty(polygons_are_empty::<false>(&polygons.into()));
            multi_polygons_with_kept_parts(parts, &keep)
        }
    })
}

/// One chunk of `base` (of `kind`) with one chunk of `parts` added.
/// `parts` holds one part per row, or with `listed=true` a list of them.
fn chunk_with_added_parts(
    kind: Kind,
    base: &dyn Array,
    parts: &dyn Array,
    listed: bool,
    kwargs: AddPartsKwargs,
) -> PolarsResult<Box<dyn Array>> {
    let rows = |what| downcast(base, what);
    Ok(match kind {
        Kind::Point | Kind::MultiPoint => {
            let base = match kind {
                Kind::Point => one_point_per_row(PointArray::new(Coords::new(base)?)),
                _ => MultiPointArray::new(rows("a list of coordinates")?)?,
            };
            let parts = point_parts(parts, listed, kwargs)?;
            let validity = rows_validity(&base.rows, &parts.rows, kwargs.null_behavior);
            let added = multi_points_with_added_parts(&base, &parts, validity);
            added.into_list_array().boxed()
        }
        Kind::LineString | Kind::MultiLineString => {
            let base = match kind {
                Kind::LineString => {
                    one_line_per_row(LineStringArray::new(rows("a list of coordinates")?)?)
                }
                _ => MultiLineStringArray::new(rows("a list of linestrings")?)?,
            };
            let parts = line_parts(parts, listed, kwargs)?;
            let validity = rows_validity(&base.rows, &parts.rows, kwargs.null_behavior);
            let added = multi_line_strings_with_added_parts(&base, &parts, validity);
            added.into_list_array().boxed()
        }
        Kind::Polygon | Kind::MultiPolygon => {
            let base = match kind {
                Kind::Polygon => one_polygon_per_row(PolygonArray::new(rows("a list of rings")?)?),
                _ => MultiPolygonArray::new(rows("a list of polygons")?)?,
            };
            let parts = polygon_parts(parts, listed, kwargs)?;
            let validity = rows_validity(&base.rows, &parts.rows, kwargs.null_behavior);
            let added = multi_polygons_with_added_parts(&base, &parts, validity);
            added.into_list_array().boxed()
        }
    })
}

/// See `add_parts`.
#[polars_expr(output_type_func=kernel_output_type_add_parts)]
fn add_parts(inputs: &[Series], kwargs: AddPartsKwargs) -> PolarsResult<Series> {
    let (geo, listed) = parts_added(&fields_of(inputs))?;
    let kind = describe(inputs[0].dtype())?.kind;
    let base = inputs[0].ext()?.storage();
    let parts = storage_of(&inputs[1])?;
    let (base, parts) = paired_rows(base, &parts, ("geometries", "parts"))?;

    let chunk = chunk_with_added_parts(
        kind,
        base.chunks()[0].as_ref(),
        parts.chunks()[0].as_ref(),
        listed,
        kwargs,
    )?;
    // SAFETY: the chunk is the storage of a multi-geometry of `geo`'s kind and dimension.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), vec![chunk], &geo.dtype())
    })
}
