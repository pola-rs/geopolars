//! Building a geometry out of the parts it is made of.

use std::sync::Arc;

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::close_rings::CloseRingsKwargs;
use crate::geoarrow::crs::{ExtensionMetadata, MetadataKwargs};
use crate::geoarrow::{coord, describe, Geo, GeoDimension, Kind};
use geopolars_compute::inner_nulls::{
    propagate_inner_nulls_of_line_strings, propagate_inner_nulls_of_multi_line_strings,
    propagate_inner_nulls_of_multi_points, propagate_inner_nulls_of_multi_polygons,
    propagate_inner_nulls_of_polygons,
};
use geopolars_compute::native::{
    close_open_rings, multi_polygons_have_only_rings, polygons_from_exteriors_and_holes,
    polygons_have_only_rings,
};
use geopolars_compute::with_const_flags;
use geopolars_core::{
    LineStringArray, List, MultiLineStringArray, MultiPolygonArray, PolygonArray, UncheckedStorage,
    ValidCoords, ValidList,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::Offsets;

/// The keyword arguments of the constructors that build polygons:
/// [`MetadataKwargs`], plus whether and how to close open rings.
/// These have to be mirrored on the python side.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PolygonKwargs {
    /// The CRS the coordinates are in, in any form PROJ accepts.
    pub crs: Option<String>,
    /// How to close every open ring, as `close_rings` does. `None` closes nothing.
    pub close: Option<CloseRingsKwargs>,
}

impl PolygonKwargs {
    fn metadata(&self) -> MetadataKwargs {
        MetadataKwargs {
            crs: self.crs.clone(),
        }
    }
}

/// Used in error messages, don't care about allocation
fn shape_of(part: Kind) -> String {
    let lists = "lists of ".repeat(part.nesting() as usize);
    format!("{lists}coordinate structs")
}

/// Used in error messages, don't care about allocation
fn coordinate_shape_of(nesting: u8) -> String {
    let lists = "lists of ".repeat(nesting as usize);
    format!("{lists}f64")
}

/// The dimension and metadata of the parts a geometry can be gathered from:
/// a list of `part`s, each either a `part` geometry already or the bare storage
/// one of those wraps.
fn parts_of(dtype: &DataType, part: Kind) -> PolarsResult<(GeoDimension, Arc<ExtensionMetadata>)> {
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
                return Ok((geo.dim, geo.metadata.clone()));
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
    Ok((dim, Default::default()))
}

/// Small helper
fn gathered(
    dtype: &DataType,
    kwargs: &MetadataKwargs,
    kind: Kind,
    part: Kind,
) -> PolarsResult<Geo> {
    let (dim, metadata) = parts_of(dtype, part)?;
    Ok(Geo::new(kind, dim, kwargs.apply(metadata)?))
}

/// For gathering a list of `part`s into a `kind`.
fn kernel_output_type_wrapper_gathered(
    input_fields: &[Field],
    kwargs: &MetadataKwargs,
    kind: Kind,
    part: Kind,
) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = gathered(field.dtype(), kwargs, kind, part)?;
    Ok(Field::new(field.name().clone(), geo.dtype()))
}

fn line_string_of(dtype: &DataType, kwargs: &MetadataKwargs) -> PolarsResult<Geo> {
    match describe(dtype) {
        Ok(geo) if geo.kind == Kind::MultiPoint => Ok(Geo::new(
            Kind::LineString,
            geo.dim,
            kwargs.apply(geo.metadata.clone())?,
        )),
        _ => gathered(dtype, kwargs, Kind::LineString, Kind::Point),
    }
}

fn kernel_output_type_linestring(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = line_string_of(field.dtype(), &kwargs)?;
    Ok(Field::new(field.name().clone(), geo.dtype()))
}

fn kernel_output_type_polygon(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_gathered(
        input_fields,
        &kwargs.metadata(),
        Kind::Polygon,
        Kind::LineString,
    )
}

fn kernel_output_type_multipoint(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_gathered(input_fields, &kwargs, Kind::MultiPoint, Kind::Point)
}

fn kernel_output_type_multilinestring(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_gathered(
        input_fields,
        &kwargs,
        Kind::MultiLineString,
        Kind::LineString,
    )
}

fn kernel_output_type_multipolygon(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_gathered(
        input_fields,
        &kwargs.metadata(),
        Kind::MultiPolygon,
        Kind::Polygon,
    )
}

/// See [`close_open_rings`].
/// The rings and their coordinates, or `None` if no ring changed.
pub(super) fn closed_rings(
    rings: &ValidList,
    coords: &ValidCoords,
    closing: CloseRingsKwargs,
) -> Option<(ValidList, ValidCoords)> {
    let CloseRingsKwargs { snap_z, snap_m } = closing;
    let z = coords.dim().has_z();
    let closed = with_const_flags!(Z = z; close_open_rings::<Z>(rings, coords, snap_z, snap_m))?;
    Some((closed.rows.ignoring_validity().clone(), closed.coords))
}

/// `array` with its open rings closed first if there is a `close`,
/// and null where it does not have only rings (see [polygons_have_only_rings]).
fn validated_polygons(mut array: PolygonArray, close: Option<CloseRingsKwargs>) -> Box<dyn Array> {
    if let Some((rings, coords)) =
        close.and_then(|closing| closed_rings(&array.rings, &array.coords, closing))
    {
        array = PolygonArray::from_parts(array.rows, rings, coords);
    }
    // A null row is `false` in the values too.
    let z = array.coords.dim().has_z();
    let valid = with_const_flags!(Z = z; polygons_have_only_rings::<Z>(&array))
        .values()
        .clone();
    PolygonArray::from_parts(
        array.rows.with_validity(Some(valid)),
        array.rings,
        array.coords,
    )
    .into_list_array()
    .boxed()
}

/// One chunk of the storage of a `kind`, with every null value below a row moved out to it.
/// A polygon is also null where it does not have only rings (see [polygons_have_only_rings]).
/// With a `close`, the open rings of a polygon are closed first.
fn validated_chunk(
    chunk: &dyn Array,
    kind: Kind,
    close: Option<CloseRingsKwargs>,
) -> PolarsResult<Box<dyn Array>> {
    let storage = UncheckedStorage::new(chunk)?;
    Ok(match kind {
        Kind::Point => unreachable!("constructors always gather into lists"),
        Kind::LineString => propagate_inner_nulls_of_line_strings(storage)
            .into_list_array()
            .boxed(),
        Kind::MultiPoint => propagate_inner_nulls_of_multi_points(storage)
            .into_list_array()
            .boxed(),
        Kind::MultiLineString => propagate_inner_nulls_of_multi_line_strings(storage)
            .into_list_array()
            .boxed(),
        Kind::Polygon => validated_polygons(propagate_inner_nulls_of_polygons(storage), close),
        Kind::MultiPolygon => {
            let mut array = propagate_inner_nulls_of_multi_polygons(storage);
            if let Some((rings, coords)) =
                close.and_then(|closing| closed_rings(&array.rings, &array.coords, closing))
            {
                array = MultiPolygonArray::from_parts(array.rows, array.polygons, rings, coords);
            }
            // A null row is `false` in the values too.
            let z = array.coords.dim().has_z();
            let valid = with_const_flags!(Z = z; multi_polygons_have_only_rings::<Z>(&array))
                .values()
                .clone();
            MultiPolygonArray::from_parts(
                array.rows.with_validity(Some(valid)),
                array.polygons,
                array.rings,
                array.coords,
            )
            .into_list_array()
            .boxed()
        }
    })
}

/// [validated_chunk] for every chunk of `storage`.
fn validated(storage: Series, kind: Kind, close: Option<CloseRingsKwargs>) -> PolarsResult<Series> {
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| validated_chunk(chunk.as_ref(), kind, close))
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: the kind and dimension stay the same, so every chunk is still the storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    })
}

/// Gather a list of `part`s into one geometry of `kind` per row.
/// `close` says how to close open rings, and only means something for kinds that have them.
fn gather(
    parts: &Series,
    kwargs: &MetadataKwargs,
    close: Option<CloseRingsKwargs>,
    kind: Kind,
    part: Kind,
) -> PolarsResult<Series> {
    let geo = gathered(parts.dtype(), kwargs, kind, part)?;
    // This only changes dtype, nothing on the data.
    let storage = parts.list()?.apply_to_inner(&|part| {
        Ok(match part.dtype() {
            DataType::Extension(_, _) => part.ext()?.storage().clone(),
            _ => part,
        })
    })?;

    Ok(validated(storage.into_series(), kind, close)?.into_extension(geo.instance()))
}

/// See `linestring_from_vertices`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_linestring)]
fn linestring(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    let vertices = &inputs[0];
    let geo = line_string_of(vertices.dtype(), &kwargs)?;
    match vertices.dtype() {
        // Only a multipoint gets past `line_string_of`, and its storage needs no change.
        DataType::Extension(..) => Ok(vertices
            .ext()?
            .storage()
            .clone()
            .into_extension(geo.instance())),
        _ => gather(vertices, &kwargs, None, Kind::LineString, Kind::Point),
    }
}

/// See `polygons_from_rings_in_single_column`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon)]
fn polygon(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    gather(
        &inputs[0],
        &kwargs.metadata(),
        kwargs.close,
        Kind::Polygon,
        Kind::LineString,
    )
}

/// See `multipoint_from_points`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipoint)]
fn multipoint(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    // Note that this is basically the same as linestring, but just a different [Kind].
    gather(&inputs[0], &kwargs, None, Kind::MultiPoint, Kind::Point)
}

/// See `multilinestring_from_linestrings`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multilinestring)]
fn multilinestring(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    gather(
        &inputs[0],
        &kwargs,
        None,
        Kind::MultiLineString,
        Kind::LineString,
    )
}

/// See `multipolygon_from_polygons`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipolygon)]
fn multipolygon(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    gather(
        &inputs[0],
        &kwargs.metadata(),
        kwargs.close,
        Kind::MultiPolygon,
        Kind::Polygon,
    )
}

/// The dimension a set of separate coordinate columns spells out,
/// for a `kind` whose coordinates they nest `nesting` `List` layers deep.
fn coordinates_of(input_fields: &[Field], kind: Kind, nesting: u8) -> PolarsResult<GeoDimension> {
    let mut coordinates = Vec::with_capacity(input_fields.len());
    for field in input_fields {
        let mut dtype = field.dtype();
        for _ in 0..nesting {
            let DataType::List(inner) = dtype else {
                polars_bail!(
                    SchemaMismatch: "expected `{}` to hold {}, one per {}, got: {}",
                    field.name(), coordinate_shape_of(nesting), kind.display(), field.dtype()
                );
            };
            dtype = inner;
        }
        // The values themselves are cast, but only from something that is
        // already a number: parsing coordinates out of strings is not our job.
        polars_ensure!(
            dtype.is_primitive_numeric(),
            SchemaMismatch: "expected `{}` to hold {}, got: {}",
            field.name(), coordinate_shape_of(nesting), field.dtype()
        );
        coordinates.push(Field::new(field.name().clone(), DataType::Float64));
    }

    coord::dimension_of(&DataType::Struct(coordinates)).ok_or_else(|| {
        polars_err!(
            SchemaMismatch: "expected coordinate columns named x, y[, z][, m], in that order"
        )
    })
}

/// Small helper
fn zipped(input_fields: &[Field], kwargs: &MetadataKwargs, kind: Kind) -> PolarsResult<Geo> {
    let dim = coordinates_of(input_fields, kind, kind.nesting())?;
    Ok(Geo::new(kind, dim, kwargs.apply(Default::default())?))
}

/// The polygons or multipolygons `polygon_coords` and `multipolygon_coords` zip:
/// one ring per polygon, so nested one layer less deep than a `kind`.
fn zipped_exteriors(
    input_fields: &[Field],
    kwargs: &MetadataKwargs,
    kind: Kind,
) -> PolarsResult<Geo> {
    let dim = coordinates_of(input_fields, kind, kind.nesting() - 1)?;
    Ok(Geo::new(kind, dim, kwargs.apply(Default::default())?))
}

/// For zipping coordinate columns into a `kind`.
fn kernel_output_type_wrapper_zipped(
    input_fields: &[Field],
    kwargs: &MetadataKwargs,
    kind: Kind,
) -> PolarsResult<Field> {
    let geo = zipped(input_fields, kwargs, kind)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

fn kernel_output_type_linestring_coords(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_zipped(input_fields, &kwargs, Kind::LineString)
}

fn kernel_output_type_polygon_coords(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    let geo = zipped_exteriors(input_fields, &kwargs.metadata(), Kind::Polygon)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

fn kernel_output_type_polygon_nested_coords(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_zipped(input_fields, &kwargs.metadata(), Kind::Polygon)
}

fn kernel_output_type_multipoint_coords(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_zipped(input_fields, &kwargs, Kind::MultiPoint)
}

fn kernel_output_type_multilinestring_coords(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_zipped(input_fields, &kwargs, Kind::MultiLineString)
}

fn kernel_output_type_multipolygon_coords(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    let geo = zipped_exteriors(input_fields, &kwargs.metadata(), Kind::MultiPolygon)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

fn kernel_output_type_multipolygon_nested_coords(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_zipped(input_fields, &kwargs.metadata(), Kind::MultiPolygon)
}

/// Interleave columns that each nest their coordinate `nesting` `List` layers
/// deep into one `List`-of-that-depth of coordinate structs.
///
/// Only the first column's offsets survive; the callers checks the rest carry the same ones.
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

fn fields_of(inputs: &[Series]) -> Vec<Field> {
    inputs
        .iter()
        .map(|s| Field::new(s.name().clone(), s.dtype().clone()))
        .collect()
}

/// Zip one coordinate column per axis into one geometry of `kind` per row.
/// `close` says how to close open rings, and only means something for kinds that have them.
fn zip(
    inputs: &[Series],
    kwargs: &MetadataKwargs,
    close: Option<CloseRingsKwargs>,
    kind: Kind,
) -> PolarsResult<Series> {
    let geo = zipped(&fields_of(inputs), kwargs, kind)?;
    let storage = interleaved_columns(inputs, kind.nesting())?;
    Ok(validated(storage, kind, close)?.into_extension(geo.instance()))
}

/// The storage one coordinate column per axis zips into,
/// with the coordinates nested `nesting` `List` layers deep.
fn interleaved_columns(inputs: &[Series], nesting: u8) -> PolarsResult<Series> {
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
            differ in length, or in which of them are null",
            columns[0].name(), column.name()
        );
    }

    interleave(&columns, nesting)
}

/// See `linestring_from_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_linestring_coords)]
fn linestring_coords(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs, None, Kind::LineString)
}

/// See `polygons_from_coordinates`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon_coords)]
fn polygon_coords(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    let geo = zipped_exteriors(&fields_of(inputs), &kwargs.metadata(), Kind::Polygon)?;
    let exteriors = interleaved_columns(inputs, Kind::LineString.nesting())?;
    let chunks = exteriors
        .chunks()
        .iter()
        .map(|chunk| assembled_polygons(chunk.as_ref(), None, kwargs.close))
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: every chunk is the storage of a polygon of `geo`'s dimension.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(exteriors.name().clone(), chunks, &geo.dtype())
    })
}

/// See `polygons_from_nested_coordinates`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon_nested_coords)]
fn polygon_nested_coords(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs.metadata(), kwargs.close, Kind::Polygon)
}

/// See `multipoint_from_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipoint_coords)]
fn multipoint_coords(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs, None, Kind::MultiPoint)
}

/// See `multilinestring_from_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multilinestring_coords)]
fn multilinestring_coords(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs, None, Kind::MultiLineString)
}

/// `rows` of rings, as the rows of polygons of one ring each.
fn one_ring_per_polygon(rows: &ListArray<i64>) -> ListArray<i64> {
    let rings = rows.values().clone();
    let lengths = std::iter::repeat_n(1, rings.len());
    let offsets = Offsets::<i64>::try_from_lengths(lengths).expect("one per polygon");
    let polygon_dtype = ListArray::<i64>::default_datatype(rings.dtype().clone());
    let polygons = ListArray::new(polygon_dtype.clone(), offsets.into(), rings, None);
    ListArray::new(
        ListArray::<i64>::default_datatype(polygon_dtype),
        rows.offsets().clone(),
        polygons.boxed(),
        rows.validity().cloned(),
    )
}

/// See `multi_polygons_from_coordinates`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipolygon_coords)]
fn multipolygon_coords(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    let geo = zipped_exteriors(&fields_of(inputs), &kwargs.metadata(), Kind::MultiPolygon)?;
    let exteriors = interleaved_columns(inputs, Kind::MultiLineString.nesting())?;
    let chunks = exteriors
        .list()?
        .downcast_iter()
        .map(|chunk| {
            let multi_polygons = one_ring_per_polygon(chunk);
            validated_chunk(&multi_polygons, Kind::MultiPolygon, kwargs.close)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: every chunk is the storage of a multipolygon of `geo`'s dimension.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(exteriors.name().clone(), chunks, &geo.dtype())
    })
}

/// See `multi_polygons_from_nested_coordinates`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipolygon_nested_coords)]
fn multipolygon_nested_coords(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs.metadata(), kwargs.close, Kind::MultiPolygon)
}

fn ring_of(dtype: &DataType) -> PolarsResult<(GeoDimension, Arc<ExtensionMetadata>)> {
    if !matches!(dtype, DataType::Extension(..)) {
        return parts_of(dtype, Kind::Point);
    }
    match describe(dtype) {
        Ok(geo) if geo.kind == Kind::LineString => Ok((geo.dim, geo.metadata.clone())),
        _ => polars_bail!(
            SchemaMismatch: "expected a ring, as a `linestring` or a list of vertices, got: {}",
            dtype
        ),
    }
}

fn is_list_of_rings(dtype: &DataType) -> bool {
    let DataType::List(inner) = dtype else {
        return false;
    };
    match inner.as_ref() {
        DataType::List(_) => true,
        DataType::Extension(..) => describe(inner).is_ok_and(|geo| geo.kind == Kind::LineString),
        _ => false,
    }
}

/// The polygons `polygon_rings_in_separate_columns` builds,
/// and whether the holes (if passed) are a list of rings per row.
fn separate_rings(input_fields: &[Field], kwargs: &MetadataKwargs) -> PolarsResult<(Geo, bool)> {
    let (dim, metadata) = ring_of(input_fields[0].dtype())?;
    let Some(holes) = input_fields.get(1) else {
        return Ok((Geo::new(Kind::Polygon, dim, kwargs.apply(metadata)?), false));
    };

    let listed = is_list_of_rings(holes.dtype());
    let (hole_dim, hole_metadata) = match holes.dtype() {
        DataType::List(inner) if listed => ring_of(inner)?,
        dtype => ring_of(dtype)?,
    };
    polars_ensure!(
        hole_dim == dim,
        SchemaMismatch: "the exterior is {} but the holes are {}", dim.tag(), hole_dim.tag()
    );
    if metadata.declares_crs() && hole_metadata.declares_crs() {
        let (crs, hole_crs) = (metadata.crs()?, hole_metadata.crs()?);
        polars_ensure!(
            crs == hole_crs,
            SchemaMismatch: "the exterior and the holes declare different CRSs, {crs} and {hole_crs}; \
            use `to_crs` to bring one onto the other first"
        );
    }
    Ok((
        Geo::new(Kind::Polygon, dim, kwargs.apply(metadata)?),
        listed,
    ))
}

fn kernel_output_type_polygon_rings_in_separate_columns(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    let (geo, _) = separate_rings(input_fields, &kwargs.metadata())?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

fn storage_of(series: &Series) -> PolarsResult<Series> {
    Ok(match series.dtype() {
        DataType::Extension(..) => series.ext()?.storage().clone(),
        DataType::List(_) => series
            .list()?
            .apply_to_inner(&|inner| storage_of(&inner))?
            .into_series(),
        _ => series.clone(),
    })
}

fn one_hole_per_row(holes: LineStringArray) -> MultiLineStringArray {
    let lengths = std::iter::repeat_n(1, holes.rows.len());
    let offsets = Offsets::try_from_lengths(lengths).expect("one per row");
    MultiLineStringArray::from_parts(
        List::from_offsets(offsets.into(), holes.rows.validity().cloned()),
        holes.rows.ignoring_validity().clone(),
        holes.coords,
    )
}

/// Which polygons are there: null where the exterior is,
/// or where the holes of the row were given but something in them was null.
/// `given` is the validity of the rows of holes before inner nulls moved out to them.
fn assembled_validity(
    exteriors: &LineStringArray,
    holes: &MultiLineStringArray,
    given: Option<&Bitmap>,
) -> Option<Bitmap> {
    let lost = holes.rows.validity().map(|kept| match given {
        Some(given) => given & &!kept,
        None => !kept,
    });
    let not_lost = lost.map(|lost| !&lost);
    combine_validities_and(exteriors.rows.validity(), not_lost.as_ref())
}

/// One chunk of polygons out of one chunk of exterior rings, and of holes.
/// `holes` holds one ring per row, or with `listed=true` a list of them.
/// See `polygons_from_rings_in_separate_columns` for the null values.
fn assembled_polygons(
    exteriors: &dyn Array,
    holes: Option<(&dyn Array, bool)>,
    close: Option<CloseRingsKwargs>,
) -> PolarsResult<Box<dyn Array>> {
    let exteriors = propagate_inner_nulls_of_line_strings(UncheckedStorage::new(exteriors)?);
    let Some((holes, listed)) = holes else {
        let validity = exteriors.rows.validity().cloned();
        let polygons = polygons_from_exteriors_and_holes(&exteriors, None, validity);
        return Ok(validated_polygons(polygons, close));
    };

    let storage = UncheckedStorage::new(holes)?;
    let given = storage.lists[0].validity().cloned();
    let holes = if listed {
        propagate_inner_nulls_of_multi_line_strings(storage)
    } else {
        one_hole_per_row(propagate_inner_nulls_of_line_strings(storage))
    };
    let validity = assembled_validity(&exteriors, &holes, given.as_ref());
    let polygons = polygons_from_exteriors_and_holes(&exteriors, Some(&holes), validity);
    Ok(validated_polygons(polygons, close))
}

/// See `polygons_from_rings_in_separate_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon_rings_in_separate_columns)]
fn polygon_rings_in_separate_columns(
    inputs: &[Series],
    kwargs: PolygonKwargs,
) -> PolarsResult<Series> {
    let (geo, listed) = separate_rings(&fields_of(inputs), &kwargs.metadata())?;
    let exteriors = storage_of(&inputs[0])?.rechunk();
    let holes = inputs
        .get(1)
        .map(|holes| Ok::<_, PolarsError>(storage_of(holes)?.rechunk()))
        .transpose()?;
    if let Some(holes) = &holes {
        polars_ensure!(
            holes.len() == exteriors.len(),
            ShapeMismatch: "the exterior has {} rows but the holes have {}",
            exteriors.len(), holes.len()
        );
    }

    let holes_chunk = holes
        .as_ref()
        .map(|holes| (holes.chunks()[0].as_ref(), listed));
    let chunk = assembled_polygons(exteriors.chunks()[0].as_ref(), holes_chunk, kwargs.close)?;
    // SAFETY: the chunk is the storage of a polygon of `geo`'s dimension.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), vec![chunk], &geo.dtype())
    })
}
