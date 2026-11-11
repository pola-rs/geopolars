//! Converting between `geoarrow.wkb` and our native geometries.
//!
//! The `wkb` crate does the encoding and decoding itself.

use ::wkb::reader::read_wkb;
use ::wkb::writer::{write_geometry, WriteOptions};
use geopolars_core::{GeometryArray, GeometryBuilder};
use polars::prelude::*;
use polars_arrow::array::MutableBinaryViewArray;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::decode::{target, DecodeKwargs};
use crate::geoarrow::describe;
use crate::geoarrow::encoded::{describe_encoded, Encoded, Encoding};

/// For [`to_wkb`]: WKB carrying the geometry's metadata.
fn kernel_output_type_to_wkb(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = describe(field.dtype())?;
    let wkb = Encoded::new(Encoding::Wkb, geo.metadata.clone());
    Ok(Field::new(field.name().clone(), wkb.dtype()))
}

/// See `to_wkb`.
#[polars_expr(output_type_func=kernel_output_type_to_wkb)]
fn to_wkb(inputs: &[Series]) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let options = WriteOptions::default();

    let mut out = MutableBinaryViewArray::<[u8]>::with_capacity(storage.len());
    let mut buf = Vec::new();
    for chunk in storage.chunks() {
        GeometryArray::new(chunk.as_ref(), geo.kind)?.for_each_geometry(|geometry| {
            let Some(geometry) = geometry else {
                out.push_null();
                return Ok(());
            };
            buf.clear();
            write_geometry(&mut buf, &geometry, &options)
                .map_err(|e| polars_err!(ComputeError: "failed to write WKB: {e}"))?;
            out.push_value(&buf);
            Ok(())
        })?;
    }

    let binary = BinaryChunked::with_chunk(inputs[0].name().clone(), out.freeze()).into_series();
    let wkb = Encoded::new(Encoding::Wkb, geo.metadata.clone());
    Ok(binary.into_extension(wkb.instance()))
}

fn kernel_output_type_from_wkb(
    input_fields: &[Field],
    kwargs: DecodeKwargs,
) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = target(describe_encoded(field.dtype(), Encoding::Wkb)?, &kwargs)?;
    Ok(Field::new(field.name().clone(), geo.dtype()))
}

/// See `from_wkb`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_from_wkb)]
fn from_wkb(inputs: &[Series], kwargs: DecodeKwargs) -> PolarsResult<Series> {
    let geo = target(describe_encoded(inputs[0].dtype(), Encoding::Wkb)?, &kwargs)?;
    let binary = match inputs[0].dtype() {
        DataType::Extension(..) => inputs[0].ext()?.storage().clone(),
        _ => inputs[0].clone(),
    };

    let mut builder =
        GeometryBuilder::new(geo.kind(), geo.dim(), Encoding::Wkb.label(), binary.len());
    for value in binary
        .binary()?
        .downcast_iter()
        .flat_map(|chunk| chunk.iter())
    {
        match value {
            None => builder.null(),
            Some(bytes) => {
                let wkb =
                    read_wkb(bytes).map_err(|e| polars_err!(ComputeError: "invalid WKB: {e}"))?;
                builder.geometry(&wkb)?;
            }
        }
    }

    let storage = Series::from_arrow(inputs[0].name().clone(), builder.finish()?)?;
    Ok(storage.into_extension(geo.instance()))
}
