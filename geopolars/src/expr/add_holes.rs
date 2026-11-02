//! Adding holes to polygons, from another column.
//! See `add_holes` in Python for the rules.

use geopolars_compute::native::polygons_with_added_holes;
use geopolars_core::{MultiLineStringArray, PolygonArray};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::close_rings::CloseRingsKwargs;
use super::construct::{
    assembled_validity, fields_of, holes_per_row, listed_holes, storage_of, validated_polygons,
};
use crate::geoarrow::{describe, Geo, Kind};

/// The keyword arguments of `add_holes`.
/// These have to be mirrored on the python side.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AddHolesKwargs {
    /// How to close every open ring, as `close_rings` does. `None` closes nothing.
    pub close: Option<CloseRingsKwargs>,
}

/// The polygons `add_holes` gives back,
/// and whether the holes are a list of rings per row.
fn holes_added(input_fields: &[Field]) -> PolarsResult<(Geo, bool)> {
    let dtype = input_fields[0].dtype();
    let geo = describe(dtype)?;
    polars_ensure!(
        geo.kind == Kind::Polygon,
        SchemaMismatch: "add_holes expects a polygon column, got: {}", dtype
    );
    let listed = listed_holes(&input_fields[1], "polygon", geo.dim, geo.metadata)?;
    Ok((
        Geo::new(Kind::Polygon, geo.dim, geo.metadata.clone()),
        listed,
    ))
}

fn kernel_output_type_add_holes(input_fields: &[Field]) -> PolarsResult<Field> {
    let (geo, _) = holes_added(input_fields)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

/// Where a polygon has room for its holes:
/// everywhere but a polygon without rings that gets some.
/// Without this, its first hole would become its exterior, and pass validation.
fn has_room_for_holes(polygons: &PolygonArray, holes: &MultiLineStringArray) -> Bitmap {
    (0..polygons.rows.len())
        .map(|i| {
            !polygons.rows.range(i).is_empty()
                || !holes.rows.is_valid(i)
                || holes.rows.range(i).is_empty()
        })
        .collect()
}

/// One chunk of `polygons` with one chunk of `holes` added.
/// `holes` holds one ring per row, or with `listed=true` a list of them.
/// See `add_holes` for the null values.
fn chunk_with_added_holes(
    polygons: &ListArray<i64>,
    holes: &dyn Array,
    listed: bool,
    close: Option<CloseRingsKwargs>,
) -> PolarsResult<Box<dyn Array>> {
    let polygons = PolygonArray::new(polygons)?;
    let (holes, given) = holes_per_row(holes, listed)?;
    let validity = assembled_validity(polygons.rows.validity(), &holes, given.as_ref());
    let room = has_room_for_holes(&polygons, &holes);
    let validity = combine_validities_and(validity.as_ref(), Some(&room));
    let added = polygons_with_added_holes(&polygons, Some(&holes), validity);
    Ok(validated_polygons(added, close))
}

/// See `add_holes`.
#[polars_expr(output_type_func=kernel_output_type_add_holes)]
fn add_holes(inputs: &[Series], kwargs: AddHolesKwargs) -> PolarsResult<Series> {
    let (geo, listed) = holes_added(&fields_of(inputs))?;
    let polygons = inputs[0].ext()?.storage();
    let holes = storage_of(&inputs[1])?;
    let (polygons, holes) = match (polygons.len(), holes.len()) {
        (n, m) if n == m => (polygons.rechunk(), holes.rechunk()),
        (n, 1) => (polygons.rechunk(), holes.new_from_index(0, n).rechunk()),
        (1, m) => (polygons.new_from_index(0, m).rechunk(), holes.rechunk()),
        (n, m) => polars_bail!(ShapeMismatch: "the polygons have {n} rows but the holes have {m}"),
    };

    let rows = polygons.list()?.downcast_as_array();
    let chunk = chunk_with_added_holes(rows, holes.chunks()[0].as_ref(), listed, kwargs.close)?;
    // SAFETY: the chunk is the storage of a polygon of `geo`'s dimension.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), vec![chunk], &geo.dtype())
    })
}
