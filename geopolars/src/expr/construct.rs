//! Building a geometry out of the parts it is made of.

use std::borrow::Cow;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use super::coords::same_geometry;
use crate::geoarrow::{coord, describe, Dimension, Geo, Kind};

/// Used in error messages, don't care about allocation
fn shape_of(part: Kind) -> String {
    let lists = "lists of ".repeat(part.nesting() as usize);
    format!("{lists}coordinate structs")
}

/// Used in error messages, don't care about allocation
fn coordinate_shape_of(kind: Kind) -> String {
    let lists = "lists of ".repeat(kind.nesting() as usize);
    format!("{lists}f64")
}

/// The dimension and metadata of the parts a geometry can be gathered from:
/// a list of `part`s, each either a `part` geometry already or the bare storage
/// one of those wraps.
fn parts_of(dtype: &DataType, part: Kind) -> PolarsResult<(Dimension, Option<String>)> {
    let expected = format!(
        "expected a list of `{}`s or of {}",
        part.name(),
        shape_of(part)
    );

    let DataType::List(inner) = dtype else {
        polars_bail!(SchemaMismatch: "{}, got: {}", expected, dtype);
    };

    if matches!(inner.as_ref(), DataType::Extension(..)) {
        match describe(inner) {
            Ok(geo) if geo.kind == part => {
                // metadata should be carried over
                return Ok((geo.dim, geo.typ.serialize_metadata().map(Cow::into_owned)));
            }
            _ => polars_bail!(
                SchemaMismatch: "{} with separated x/y[/z][/m] coordinates, got: {}",
                expected, dtype
            ),
        }
    }

    let Some(dim) = coord::dimension_of_storage(inner, part.nesting()) else {
        polars_bail!(SchemaMismatch: "{}, got: {}", expected, dtype);
    };
    Ok((dim, None))
}

/// `output_type_func` for gathering a list of `part`s into a `kind`.
fn gathered_type(input_fields: &[Field], kind: Kind, part: Kind) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let (dim, metadata) = parts_of(field.dtype(), part)?;
    Ok(Field::new(
        field.name().clone(),
        Geo::new(kind, dim, metadata).dtype(),
    ))
}

fn linestring_type(input_fields: &[Field]) -> PolarsResult<Field> {
    gathered_type(input_fields, Kind::LineString, Kind::Point)
}

fn polygon_type(input_fields: &[Field]) -> PolarsResult<Field> {
    gathered_type(input_fields, Kind::Polygon, Kind::LineString)
}

fn multipoint_type(input_fields: &[Field]) -> PolarsResult<Field> {
    gathered_type(input_fields, Kind::MultiPoint, Kind::Point)
}

/// Elementwise: is this geometry there in full, down to the last coordinate?
///
/// A part that is missing anywhere below the outermost level
/// (e.g. a vertex of a linestring,
/// a ring of a polygon,
/// a vertex of one of those rings)
/// counts as missing here rather than as null,
/// because the geometry around it cannot stand without it.
fn complete(values: &Series, nesting: u8) -> PolarsResult<BooleanChunked> {
    if nesting == 0 {
        let coords = values.struct_()?;
        return Ok(coords
            .fields_as_series()
            .iter()
            .fold(values.is_not_null(), |present, coord| {
                present & coord.is_not_null()
            }));
    }

    let parts = values.list()?;
    let inner = complete(&parts.get_inner(), nesting - 1)?;
    // Borrow the offsets to ask the same question one level further out.
    parts
        .with_inner_values(&inner.into_series())
        .amortized_iter()
        .map(|part| match part {
            // At the outermost level this says "incomplete" of a row that is
            // already null, which nulling it again leaves alone.
            None => Ok(false),
            Some(part) => Ok(part.as_ref().bool()?.all()),
        })
        .collect()
}

/// Null out every geometry that is missing a part, or a coordinate of one.
///
/// GeoArrow allows nulls only at the outermost level.
fn only_complete(parts: Series, nesting: u8) -> PolarsResult<Series> {
    let complete = complete(&parts, nesting)?;
    if complete.all() {
        return Ok(parts);
    }

    let missing = Series::full_null(parts.name().clone(), parts.len(), parts.dtype());
    parts.zip_with(&complete, &missing)
}

/// Gather a list of `part`s into one geometry of `kind` per row.
fn gather(parts: &Series, kind: Kind, part: Kind) -> PolarsResult<Series> {
    let (dim, metadata) = parts_of(parts.dtype(), part)?;
    // This only changes dtype, nothing on the data.
    let storage = parts.list()?.apply_to_inner(&|part| {
        Ok(match part.dtype() {
            DataType::Extension(_, _) => part.ext()?.storage().clone(),
            _ => part,
        })
    })?;

    Ok(only_complete(storage.into_series(), kind.nesting())?
        .into_extension(Geo::new(kind, dim, metadata).instance()))
}

/// Gather lists of vertices into linestrings.
#[polars_expr(output_type_func=linestring_type)]
fn linestring(inputs: &[Series]) -> PolarsResult<Series> {
    gather(&inputs[0], Kind::LineString, Kind::Point)
}

/// Gather lists of rings into polygons.
#[polars_expr(output_type_func=polygon_type)]
fn polygon(inputs: &[Series]) -> PolarsResult<Series> {
    gather(&inputs[0], Kind::Polygon, Kind::LineString)
}

/// Gather lists of points into multipoints.
///
/// The same gather a linestring is built by, over the same parts. Only the
/// `Kind` it is labelled with differs, and that is the whole difference between
/// the two geometries.
#[polars_expr(output_type_func=multipoint_type)]
fn multipoint(inputs: &[Series]) -> PolarsResult<Series> {
    gather(&inputs[0], Kind::MultiPoint, Kind::Point)
}

/// Null out every geometry that is missing a part, or a coordinate of one.
/// See python doc
#[polars_expr(output_type_func=same_geometry)]
fn validate(inputs: &[Series]) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage().clone();

    Ok(only_complete(storage, geo.kind.nesting())?.into_extension(geo.typ.clone()))
}

/// The dimension a set of separate coordinate columns spells out.
fn coordinates_of(input_fields: &[Field], kind: Kind) -> PolarsResult<Dimension> {
    let mut coordinates = Vec::with_capacity(input_fields.len());
    for field in input_fields {
        let mut dtype = field.dtype();
        for _ in 0..kind.nesting() {
            let DataType::List(inner) = dtype else {
                polars_bail!(
                    SchemaMismatch: "expected `{}` to hold {}, one per {}, got: {}",
                    field.name(), coordinate_shape_of(kind), kind.display(), field.dtype()
                );
            };
            dtype = inner;
        }
        // The values themselves are cast, but only from something that is
        // already a number: parsing coordinates out of strings is not our job.
        polars_ensure!(
            dtype.is_primitive_numeric(),
            SchemaMismatch: "expected `{}` to hold {}, got: {}",
            field.name(), coordinate_shape_of(kind), field.dtype()
        );
        coordinates.push(Field::new(field.name().clone(), DataType::Float64));
    }

    coord::dimension_of(&DataType::Struct(coordinates)).ok_or_else(|| {
        polars_err!(
            SchemaMismatch: "expected coordinate columns named x, y[, z][, m], in that order"
        )
    })
}

/// `output_type_func` for zipping coordinate columns into a `kind`.
fn zipped_type(input_fields: &[Field], kind: Kind) -> PolarsResult<Field> {
    let dim = coordinates_of(input_fields, kind)?;
    Ok(Field::new(
        input_fields[0].name().clone(),
        Geo::new(kind, dim, None).dtype(),
    ))
}

fn linestring_coords_type(input_fields: &[Field]) -> PolarsResult<Field> {
    zipped_type(input_fields, Kind::LineString)
}

fn polygon_coords_type(input_fields: &[Field]) -> PolarsResult<Field> {
    zipped_type(input_fields, Kind::Polygon)
}

fn multipoint_coords_type(input_fields: &[Field]) -> PolarsResult<Field> {
    zipped_type(input_fields, Kind::MultiPoint)
}

/// Interleave columns that each nest their coordinate `nesting` `List` layers
/// deep into one `List`-of-that-depth of coordinate structs.
///
/// Only the first column's offsets survive; the callers checks the rest carry
/// the same ones.
fn interleave(columns: &[Series], nesting: u8) -> PolarsResult<Series> {
    if nesting == 0 {
        let coordinates = columns
            .iter()
            .map(|values| values.cast(&DataType::Float64))
            .collect::<PolarsResult<Vec<Series>>>()?;
        let name = columns[0].name().clone();
        let len = columns[0].len();
        return Ok(StructChunked::from_series(name, len, coordinates.iter())?.into_series());
    }

    let inner = columns
        .iter()
        .map(|column| Ok(column.list()?.get_inner()))
        .collect::<PolarsResult<Vec<Series>>>()?;
    Ok(columns[0]
        .list()?
        .with_inner_values(&interleave(&inner, nesting - 1)?)
        .into_series())
}

/// Zip one coordinate column per axis into one geometry of `kind` per row.
fn zip(inputs: &[Series], kind: Kind) -> PolarsResult<Series> {
    let input_fields: Vec<Field> = inputs
        .iter()
        .map(|s| Field::new(s.name().clone(), s.dtype().clone()))
        .collect();
    let dim = coordinates_of(&input_fields, kind)?;

    // `get_inner` reads the values array whole, so anything a slice left
    // outside the offsets has to go before the columns can be read in lockstep.
    let columns: Vec<Series> = inputs
        .iter()
        .map(|s| {
            s.trim_lists_to_normalized_offsets()
                .unwrap_or_else(|| s.clone())
                .rechunk()
        })
        .collect();

    // One set of offsets has to describe all of them, or the coordinates do not
    // line up into vertices at all.
    let structure = columns[0].list_offsets_and_validities_recursive();
    for column in &columns[1..] {
        polars_ensure!(
            column.list_offsets_and_validities_recursive() == structure,
            ComputeError: "`{}` and `{}` do not nest the same way: their lists \
            differ in length, or in which of them are missing",
            columns[0].name(), column.name()
        );
    }

    let storage = interleave(&columns, kind.nesting())?;
    Ok(
        only_complete(storage, kind.nesting())?
            .into_extension(Geo::new(kind, dim, None).instance()),
    )
}

/// Zip lists of coordinates into linestrings.
#[polars_expr(output_type_func=linestring_coords_type)]
fn linestring_coords(inputs: &[Series]) -> PolarsResult<Series> {
    zip(inputs, Kind::LineString)
}

/// Zip lists of lists of coordinates into polygons.
#[polars_expr(output_type_func=polygon_coords_type)]
fn polygon_coords(inputs: &[Series]) -> PolarsResult<Series> {
    zip(inputs, Kind::Polygon)
}

/// Zip lists of coordinates into multipoints.
#[polars_expr(output_type_func=multipoint_coords_type)]
fn multipoint_coords(inputs: &[Series]) -> PolarsResult<Series> {
    zip(inputs, Kind::MultiPoint)
}
