use geopolars_core::{EmptyExterior, NullParts};
use geopolars_views::Kind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::close_rings::CloseRingsKwargs;
use super::construct::{fields_of, listed_parts, paired_rows, parts_storage, NullPartsDef};
use crate::geoarrow::{describe, ensure_same_crs_unless_bare, geometry_series, Geo};

/// The keyword arguments of `add_holes`.
/// These have to be mirrored on the Python side.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AddHolesKwargs {
    /// How to close every open ring, as `close_rings` does.
    pub close: Option<CloseRingsKwargs>,
    #[serde(with = "NullPartsDef")]
    null_parts: NullParts,
    #[serde(with = "EmptyExteriorDef")]
    empty_exterior: EmptyExterior,
}

#[derive(Deserialize)]
#[serde(remote = "EmptyExterior", rename_all = "snake_case")]
enum EmptyExteriorDef {
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

/// See `add_holes`.
#[polars_expr(output_type_func=kernel_output_type_add_holes)]
fn add_holes(inputs: &[Series], kwargs: AddHolesKwargs) -> PolarsResult<Series> {
    let (geo, listed) = holes_added(&fields_of(inputs))?;
    let polygons = inputs[0].ext()?.storage();
    let holes = parts_storage(&inputs[1], Kind::MultiLineString, geo.dim())?;
    let (polygons, holes) = paired_rows(polygons, &holes, ("polygons", "holes"))?;

    let chunk = geopolars_core::add_holes(
        polygons.chunks()[0].as_ref(),
        holes.chunks()[0].as_ref(),
        listed,
        kwargs.null_parts,
        kwargs.empty_exterior,
        kwargs.close.map(Into::into),
    )?;
    geometry_series(inputs[0].name().clone(), geo, vec![chunk])
}
