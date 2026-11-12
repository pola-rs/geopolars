//! The rings of a polygon (see the `rings` module of `geopolars_compute::native`).

use geopolars_compute::native::{
    multi_polygon_exteriors, multi_polygon_interiors, polygon_exteriors, polygon_interiors,
};
use geopolars_views::{Kind, MultiPolygonArray, PolygonArray, Untrimmed};
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::{describe, Geo};

/// A `kind` per polygon, or a list of them per multipolygon,
/// with the input's dimension and metadata.
/// Also whether the input is a multipolygon.
fn rings_type(dtype: &DataType, name: &str, kind: Kind) -> PolarsResult<(DataType, bool)> {
    let geo = describe(dtype)?;
    let multi = match geo.kind {
        Kind::Polygon => false,
        Kind::MultiPolygon => true,
        _ => polars_bail!(
            SchemaMismatch: "{name} expects a polygon or multipolygon column, got: {}", dtype
        ),
    };
    let part = Geo::new(kind, geo.dim, geo.metadata.clone()).dtype();
    let dtype = if multi {
        DataType::List(Box::new(part))
    } else {
        part
    };
    Ok((dtype, multi))
}

/// For [`exterior`].
fn kernel_output_type_exterior(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = rings_type(input_fields[0].dtype(), "exterior", Kind::LineString)?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// For [`interior`].
fn kernel_output_type_interior(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = rings_type(input_fields[0].dtype(), "interior", Kind::MultiLineString)?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// Every chunk of the polygons `inputs[0]` put through `polygons`, as a Series of `dtype`.
/// If `multi=true`, they are multipolygons, put through `multi_polygons` instead.
fn per_chunk(
    inputs: &[Series],
    dtype: &DataType,
    multi: bool,
    polygons: impl Fn(&Untrimmed<PolygonArray>) -> Box<dyn Array>,
    multi_polygons: impl Fn(&Untrimmed<MultiPolygonArray>) -> Box<dyn Array>,
) -> PolarsResult<Series> {
    let storage = inputs[0].ext()?.storage();
    let chunks = storage.list()?.downcast_iter().map(|rows| {
        PolarsResult::Ok(if multi {
            multi_polygons(&MultiPolygonArray::new_untrimmed(rows)?)
        } else {
            polygons(&PolygonArray::new_untrimmed(rows)?)
        })
    });
    let chunks = chunks.collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: every chunk is the storage `dtype` describes.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, dtype) })
}

/// See `exterior`.
#[polars_expr(output_type_func=kernel_output_type_exterior)]
fn exterior(inputs: &[Series]) -> PolarsResult<Series> {
    let (dtype, multi) = rings_type(inputs[0].dtype(), "exterior", Kind::LineString)?;
    per_chunk(
        inputs,
        &dtype,
        multi,
        |array| polygon_exteriors(array).into_list_array().boxed(),
        |array| multi_polygon_exteriors(array).boxed(),
    )
}

/// See `interior`.
#[polars_expr(output_type_func=kernel_output_type_interior)]
fn interior(inputs: &[Series]) -> PolarsResult<Series> {
    let (dtype, multi) = rings_type(inputs[0].dtype(), "interior", Kind::MultiLineString)?;
    per_chunk(
        inputs,
        &dtype,
        multi,
        |array| polygon_interiors(array).into_list_array().boxed(),
        |array| multi_polygon_interiors(array).boxed(),
    )
}
