//! Adding holes to polygons, from another column.
//! See `add_holes` in Python for the rules.

use geopolars_compute::native::{have_room_for_holes, polygons_with_added_holes};
use geopolars_core::PolygonArray;
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use polars_arrow::compute::utils::combine_validities_and;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::close_rings::CloseRingsKwargs;
use super::construct::{
    fields_of, lines_per_row, listed_parts, paired_rows, parts_storage, rows_validity,
    validated_polygons, NullParts,
};
use crate::geoarrow::{describe, ensure_same_crs_unless_bare, Geo, Kind};

/// The keyword arguments of `add_holes`.
/// These have to be mirrored on the Python side.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AddHolesKwargs {
    /// How to close every open ring, as `close_rings` does.
    pub close: Option<CloseRingsKwargs>,
    null_parts: NullParts,
    empty_exterior: EmptyExterior,
}

/// `empty_exterior` of `add_holes`.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EmptyExterior {
    #[default]
    Null,
    Keep,
}

/// Checks the input fields of `add_holes`.
/// Returns the type of the result, and whether `holes` holds a list of rings per row.
fn holes_added(input_fields: &[Field]) -> PolarsResult<(Geo, bool)> {
    let dtype = input_fields[0].dtype();
    let geo = describe(dtype)?;
    polars_ensure!(
        geo.kind == Kind::Polygon,
        SchemaMismatch: "add_holes expects a polygon column, got: {}", dtype
    );
    let names = ("polygon", "holes");
    let (listed, holes_metadata) =
        listed_parts(&input_fields[1], Kind::LineString, geo.dim, names)?;
    ensure_same_crs_unless_bare(Some(geo.metadata), holes_metadata.as_deref(), names)?;
    Ok((
        Geo::new(Kind::Polygon, geo.dim, geo.metadata.clone()),
        listed,
    ))
}

fn kernel_output_type_add_holes(input_fields: &[Field]) -> PolarsResult<Field> {
    let (geo, _) = holes_added(input_fields)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

/// Adds `holes` to `polygons`, one chunk of each.
/// `holes` only holds a single ring per row if `listed=false`.
/// See `add_holes` for the null values.
fn chunk_with_added_holes(
    polygons: &ListArray<i64>,
    holes: &dyn Array,
    listed: bool,
    kwargs: AddHolesKwargs,
) -> PolarsResult<Box<dyn Array>> {
    let polygons = PolygonArray::new(polygons)?;
    let holes = lines_per_row(holes, listed, kwargs.null_parts)?;
    let validity = rows_validity(&polygons.rows, holes.rows(), kwargs.null_parts);
    let room = have_room_for_holes(&polygons, &holes);
    let (holes, validity) = match kwargs.empty_exterior {
        EmptyExterior::Null => (
            holes,
            combine_validities_and(validity.as_ref(), Some(&room)),
        ),
        EmptyExterior::Keep => (holes.only_on_rows(&room), validity),
    };
    let added = polygons_with_added_holes(&polygons, Some(&holes), validity);
    Ok(validated_polygons(added, kwargs.close))
}

/// See `add_holes`.
#[polars_expr(output_type_func=kernel_output_type_add_holes)]
fn add_holes(inputs: &[Series], kwargs: AddHolesKwargs) -> PolarsResult<Series> {
    let (geo, listed) = holes_added(&fields_of(inputs))?;
    let polygons = inputs[0].ext()?.storage();
    let holes = parts_storage(&inputs[1], Kind::MultiLineString, geo.dim())?;
    let (polygons, holes) = paired_rows(polygons, &holes, ("polygons", "holes"))?;

    let rows = polygons.list()?.downcast_as_array();
    let chunk = chunk_with_added_holes(rows, holes.chunks()[0].as_ref(), listed, kwargs)?;
    // SAFETY: the chunk is the storage of a polygon of `geo`'s dimension.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), vec![chunk], &geo.dtype())
    })
}
