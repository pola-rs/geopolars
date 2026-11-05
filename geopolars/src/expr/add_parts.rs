//! Adding parts to geometries, from another column.
//! See `add_parts` in Python for the rules.

use geopolars_compute::inner_nulls::{
    multi_line_strings_with_null_parts_marked, multi_points_with_null_parts_marked,
    multi_polygons_with_null_parts_marked, propagate_inner_nulls_of_line_strings,
    propagate_inner_nulls_of_multi_line_strings, propagate_inner_nulls_of_multi_points,
    propagate_inner_nulls_of_multi_polygons, propagate_inner_nulls_of_points,
    propagate_inner_nulls_of_polygons, NullPartsMarked,
};
use geopolars_compute::native::{
    multi_line_strings_with_added_parts, multi_points_with_added_parts,
    multi_polygons_with_added_parts,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, List, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, UncheckedStorage,
};
use polars::prelude::*;
use polars_arrow::array::Array;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::construct::{
    fields_of, one_line_per_row, one_point_per_row, one_polygon_per_row, paired_rows, storage_of,
    under_lists,
};
use crate::geoarrow::{coord, describe, ensure_same_crs_unless_bare, Geo, Kind};

/// `null_parts` of `add_parts`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum NullParts {
    Drop,
    Null,
}

/// `empty_parts` of `add_parts`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EmptyParts {
    Keep,
    Drop,
}

/// The keyword arguments of `add_parts`.
/// These have to be mirrored on the python side.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddPartsKwargs {
    null_parts: NullParts,
    empty_parts: EmptyParts,
}

/// The kind of the parts of a `kind`, and the multi-geometry of them.
fn part_kinds(kind: Kind) -> (Kind, Kind) {
    match kind {
        Kind::Point | Kind::MultiPoint => (Kind::Point, Kind::MultiPoint),
        Kind::LineString | Kind::MultiLineString => (Kind::LineString, Kind::MultiLineString),
        Kind::Polygon | Kind::MultiPolygon => (Kind::Polygon, Kind::MultiPolygon),
    }
}

/// The kinds besides points that hold the items of a `part`.
/// A multipoint holds the vertices of a linestring.
/// A linestring holds one ring of a polygon, and a multilinestring all of its rings.
/// The lists around them decide whether a row holds one part or a list of them.
fn kinds_holding_items_of(part: Kind) -> &'static [Kind] {
    match part {
        Kind::LineString => &[Kind::MultiPoint],
        Kind::Polygon => &[Kind::LineString, Kind::MultiLineString],
        _ => &[],
    }
}

/// The multi-geometries `add_parts` gives back,
/// and whether the parts are a list of them per row rather than one part.
fn parts_added(input_fields: &[Field]) -> PolarsResult<(Geo, bool)> {
    let geo = describe(input_fields[0].dtype())?;
    let (part, multi) = part_kinds(geo.kind);
    let parts = input_fields[1].dtype();
    let multi_geo = Geo::new(multi, geo.dim, geo.metadata.clone());
    // An untyped null (`pl.lit(None)`) is a null list of parts on every row.
    if matches!(parts, DataType::Null) {
        return Ok((multi_geo, true));
    }
    let refused = || {
        polars_err!(
            SchemaMismatch: "expected the parts as one `{}` per row, a list of them, or a `{}`; \
            a part can also be the list of its points (or of their bare coordinates){}, got: {}",
            part.display(), multi.display(),
            match part {
                Kind::LineString => ", or a `multipoint`",
                Kind::Polygon => ", or of its rings as linestrings, or a `multilinestring`",
                _ => "",
            },
            parts
        )
    };
    let (inner, lists) = under_lists(parts);
    let (nesting, dim, metadata) = match inner {
        DataType::Extension(..) => {
            let found = describe(inner).map_err(|_| refused())?;
            let accepted = [part, multi, Kind::Point].contains(&found.kind)
                || kinds_holding_items_of(part).contains(&found.kind);
            if !accepted {
                return Err(refused());
            }
            (found.kind.nesting(), found.dim, Some(&**found.metadata))
        }
        bare => (0, coord::dimension_of(bare).ok_or_else(refused)?, None),
    };
    ensure_same_crs_unless_bare(Some(geo.metadata), metadata, ("geometry", "parts"))?;
    let listed = match (lists + nesting).checked_sub(part.nesting()) {
        Some(0) => false,
        Some(1) => true,
        _ => return Err(refused()),
    };
    polars_ensure!(
        dim == geo.dim,
        SchemaMismatch: "the geometry is {} but the parts are {}", geo.dim.tag(), dim.tag()
    );
    Ok((multi_geo, listed))
}

fn kernel_output_type_add_parts(input_fields: &[Field]) -> PolarsResult<Field> {
    let (geo, _) = parts_added(input_fields)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

/// The rows of the result: null where `base` is,
/// and with [NullParts::Null] also where `parts` is.
fn rows_validity(base: &List, parts: &List, null_parts: NullParts) -> Option<Bitmap> {
    match null_parts {
        NullParts::Drop => base.validity().cloned(),
        NullParts::Null => combine_validities_and(base.validity(), parts.validity()),
    }
}

/// One chunk of the parts to add to points,
/// holding one point per row, or with `listed=true` a list of them.
fn point_parts(
    chunk: &dyn Array,
    listed: bool,
    kwargs: AddPartsKwargs,
) -> PolarsResult<NullPartsMarked<MultiPointArray>> {
    let storage = UncheckedStorage::new(chunk)?;
    let parts = match (listed, kwargs.null_parts) {
        (false, _) => one_point_per_row(propagate_inner_nulls_of_points(storage)).into(),
        (true, NullParts::Drop) => multi_points_with_null_parts_marked(storage),
        (true, NullParts::Null) => propagate_inner_nulls_of_multi_points(storage).into(),
    };
    Ok(match kwargs.empty_parts {
        EmptyParts::Keep => parts,
        EmptyParts::Drop => parts.without_empty_parts(),
    })
}

/// [point_parts], for linestrings.
fn line_parts(
    chunk: &dyn Array,
    listed: bool,
    kwargs: AddPartsKwargs,
) -> PolarsResult<NullPartsMarked<MultiLineStringArray>> {
    let storage = UncheckedStorage::new(chunk)?;
    let parts = match (listed, kwargs.null_parts) {
        (false, _) => one_line_per_row(propagate_inner_nulls_of_line_strings(storage)).into(),
        (true, NullParts::Drop) => multi_line_strings_with_null_parts_marked(storage),
        (true, NullParts::Null) => propagate_inner_nulls_of_multi_line_strings(storage).into(),
    };
    Ok(match kwargs.empty_parts {
        EmptyParts::Keep => parts,
        EmptyParts::Drop => parts.without_empty_parts(),
    })
}

/// [point_parts], for polygons.
fn polygon_parts(
    chunk: &dyn Array,
    listed: bool,
    kwargs: AddPartsKwargs,
) -> PolarsResult<NullPartsMarked<MultiPolygonArray>> {
    let storage = UncheckedStorage::new(chunk)?;
    let parts = match (listed, kwargs.null_parts) {
        (false, _) => one_polygon_per_row(propagate_inner_nulls_of_polygons(storage)).into(),
        (true, NullParts::Drop) => multi_polygons_with_null_parts_marked(storage),
        (true, NullParts::Null) => propagate_inner_nulls_of_multi_polygons(storage).into(),
    };
    Ok(match kwargs.empty_parts {
        EmptyParts::Keep => parts,
        EmptyParts::Drop => parts.without_empty_parts(),
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
            let validity = rows_validity(&base.rows, parts.rows(), kwargs.null_parts);
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
            let validity = rows_validity(&base.rows, parts.rows(), kwargs.null_parts);
            let added = multi_line_strings_with_added_parts(&base, &parts, validity);
            added.into_list_array().boxed()
        }
        Kind::Polygon | Kind::MultiPolygon => {
            let base = match kind {
                Kind::Polygon => one_polygon_per_row(PolygonArray::new(rows("a list of rings")?)?),
                _ => MultiPolygonArray::new(rows("a list of polygons")?)?,
            };
            let parts = polygon_parts(parts, listed, kwargs)?;
            let validity = rows_validity(&base.rows, parts.rows(), kwargs.null_parts);
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
    let parts = match inputs[1].dtype() {
        DataType::Null => {
            let storage = geo.dim().storage(geo.kind().nesting());
            Series::full_null(inputs[1].name().clone(), inputs[1].len(), &storage)
        }
        _ => storage_of(&inputs[1])?,
    };
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
