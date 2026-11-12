//! Whether a geometry has two points that coincide,
//! through `geopolars_compute::native::*_have_repeats_in_succession`.
//! See `has_repeats_in_succession` in Python for the rules.

use geopolars_compute::native::{
    line_strings_have_repeats_in_succession, multi_line_strings_have_repeats_in_succession,
    multi_points_have_repeats_in_succession, multi_polygons_have_repeats_in_succession,
    points_have_repeats_in_succession, polygons_have_repeats_in_succession, Coincidence,
};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::describe;

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HasRepeatsInSuccessionKwargs {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    unique_z: bool,
    unique_m: bool,
}

/// [`has_repeats_in_succession`] for one chunk of the storage of a `kind`.
fn chunk_has_repeats_in_succession(
    chunk: &dyn Array,
    kind: Kind,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
) -> PolarsResult<BooleanArray> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_have_repeats_in_succession(&PointArray::new(Coords::new(chunk)?)),
        Kind::LineString => line_strings_have_repeats_in_succession(
            &LineStringArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        ),
        Kind::MultiPoint => multi_points_have_repeats_in_succession(
            &MultiPointArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        ),
        Kind::MultiLineString => multi_line_strings_have_repeats_in_succession(
            &MultiLineStringArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        ),
        Kind::Polygon => polygons_have_repeats_in_succession(
            &PolygonArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        ),
        Kind::MultiPolygon => multi_polygons_have_repeats_in_succession(
            &MultiPolygonArray::new(rows()?)?,
            coincidence,
            unique_z,
            unique_m,
        ),
    })
}

/// For [`has_repeats_in_succession`]: a `bool` per geometry.
fn kernel_output_type_has_repeats_in_succession(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `has_repeats_in_succession`.
#[polars_expr(output_type_func=kernel_output_type_has_repeats_in_succession)]
fn has_repeats_in_succession(
    inputs: &[Series],
    kwargs: HasRepeatsInSuccessionKwargs,
) -> PolarsResult<Series> {
    let geometries = &inputs[0];
    let kind = describe(geometries.dtype())?.kind;
    let coincidence = Coincidence::new(kwargs.abs_tol, kwargs.rel_tol);
    let storage = geometries.ext()?.storage();
    let chunks = storage.chunks().iter().map(|chunk| {
        chunk_has_repeats_in_succession(
            chunk.as_ref(),
            kind,
            coincidence,
            kwargs.unique_z,
            kwargs.unique_m,
        )
    });
    Ok(BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)?.into_series())
}
