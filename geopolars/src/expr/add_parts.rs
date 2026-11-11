//! Adding parts to geometries, from another column.
//! See `add_parts` in Python for the rules.

use geopolars_compute::inner_nulls::MaskedParts;
use geopolars_compute::native::{
    multi_line_strings_with_added_parts, multi_points_with_added_parts,
    multi_polygons_with_added_parts, WithoutEmptyParts,
};
use geopolars_core::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::construct::{
    fields_of, lines_per_row, listed_parts, one_line_per_row, one_point_per_row,
    one_polygon_per_row, paired_rows, part_kinds, parts_storage, points_per_row, polygons_per_row,
    rows_validity, NullParts,
};
use crate::geoarrow::{describe, ensure_same_crs_unless_bare, Geo};

/// `empty_parts` of `add_parts`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EmptyParts {
    Keep,
    Drop,
}

/// The keyword arguments of `add_parts`.
/// These have to be mirrored on the Python side.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddPartsKwargs {
    null_parts: NullParts,
    empty_parts: EmptyParts,
}

/// Checks the input fields of `add_parts`.
/// Returns the type of the result, and whether `parts` holds a list of parts per row.
fn parts_added(input_fields: &[Field]) -> PolarsResult<(Geo, bool)> {
    let geo = describe(input_fields[0].dtype())?;
    let (part, multi) = part_kinds(geo.kind);
    let names = ("geometry", "parts");
    let (listed, metadata) = listed_parts(&input_fields[1], part, geo.dim, names)?;
    ensure_same_crs_unless_bare(Some(geo.metadata), metadata.as_deref(), names)?;
    Ok((Geo::new(multi, geo.dim, geo.metadata.clone()), listed))
}

fn kernel_output_type_add_parts(input_fields: &[Field]) -> PolarsResult<Field> {
    let (geo, _) = parts_added(input_fields)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

fn without_empty_parts_if<T: WithoutEmptyParts>(drop: bool, parts: T) -> T {
    if drop {
        parts.without_empty_parts()
    } else {
        parts
    }
}

/// Adds `parts` to the geometries of `kind` in `base_geometry`, one chunk of each.
/// `parts` only holds a single part per row if `listed=false`.
/// A single base geometry is turned into a Multi* geometry.
/// It is therefore also affected by `kwargs=`[EmptyParts::Drop].
fn chunk_with_added_parts(
    kind: Kind,
    base_geometry: &dyn Array,
    parts: &dyn Array,
    listed: bool,
    kwargs: AddPartsKwargs,
) -> PolarsResult<Box<dyn Array>> {
    let rows = |what| downcast(base_geometry, what);
    let drop = kwargs.empty_parts == EmptyParts::Drop;
    Ok(match kind {
        Kind::Point | Kind::MultiPoint => {
            let base = match kind {
                Kind::Point => without_empty_parts_if(
                    drop,
                    MaskedParts::from(one_point_per_row(PointArray::new(Coords::new(
                        base_geometry,
                    )?))),
                ),
                _ => MultiPointArray::new(rows("a list of coordinates")?)?.into(),
            };
            let parts = points_per_row(parts, listed, kwargs.null_parts)?;
            let parts = without_empty_parts_if(drop, parts);
            let validity = rows_validity(base.rows(), parts.rows(), kwargs.null_parts);
            let added = multi_points_with_added_parts(&base, &parts, validity);
            added.into_list_array().boxed()
        }
        Kind::LineString | Kind::MultiLineString => {
            let base = match kind {
                Kind::LineString => {
                    let lines = LineStringArray::new(rows("a list of coordinates")?)?;
                    without_empty_parts_if(drop, MaskedParts::from(one_line_per_row(lines)))
                }
                _ => MultiLineStringArray::new(rows("a list of linestrings")?)?.into(),
            };
            let parts = lines_per_row(parts, listed, kwargs.null_parts)?;
            let parts = without_empty_parts_if(drop, parts);
            let validity = rows_validity(base.rows(), parts.rows(), kwargs.null_parts);
            let added = multi_line_strings_with_added_parts(&base, &parts, validity);
            added.into_list_array().boxed()
        }
        Kind::Polygon | Kind::MultiPolygon => {
            let base = match kind {
                Kind::Polygon => {
                    let polygons = PolygonArray::new(rows("a list of rings")?)?;
                    without_empty_parts_if(drop, MaskedParts::from(one_polygon_per_row(polygons)))
                }
                _ => MultiPolygonArray::new(rows("a list of polygons")?)?.into(),
            };
            let parts = polygons_per_row(parts, listed, kwargs.null_parts)?;
            let parts = without_empty_parts_if(drop, parts);
            let validity = rows_validity(base.rows(), parts.rows(), kwargs.null_parts);
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
    let parts = parts_storage(&inputs[1], geo.kind(), geo.dim())?;
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
