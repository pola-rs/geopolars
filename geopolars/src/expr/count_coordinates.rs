//! How many coordinates a geometry is made of.
//! See `count_coordinates` in Python for the rules.

use geopolars_compute::native::{
    line_string_coordinate_counts, multi_line_string_coordinate_counts,
    multi_point_coordinate_counts, multi_polygon_coordinate_counts, point_coordinate_counts,
    polygon_coordinate_counts,
};
use geopolars_compute::with_const_flags;
use geopolars_core::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::describe;

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountCoordinatesKwargs {
    include_empty: bool,
}

/// For [`count_coordinates`]: a `u32` per geometry, of any kind.
fn kernel_output_type_count_coordinates(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::UInt32))
}

/// The kernel of `kind` on one chunk of the storage.
fn chunk_coordinate_counts(
    chunk: &dyn Array,
    kind: Kind,
    include_empty: bool,
) -> PolarsResult<PrimitiveArray<u32>> {
    Ok(match kind {
        Kind::Point => {
            let array = PointArray::new(Coords::new(chunk)?);
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                point_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
        Kind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                line_string_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                multi_point_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
        Kind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                polygon_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
        Kind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                multi_line_string_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(
                INCLUDE_EMPTY = include_empty;
                multi_polygon_coordinate_counts::<INCLUDE_EMPTY>(&array)
            )
        }
    })
}

/// See `count_coordinates`.
#[polars_expr(output_type_func=kernel_output_type_count_coordinates)]
fn count_coordinates(inputs: &[Series], kwargs: CountCoordinatesKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_coordinate_counts(chunk.as_ref(), kind, kwargs.include_empty));
    Ok(UInt32Chunked::try_from_chunk_iter(inputs[0].name().clone(), chunks)?.into_series())
}
