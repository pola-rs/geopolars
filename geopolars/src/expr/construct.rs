//! Building a geometry out of the parts it is made of.

use std::sync::Arc;

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::close_rings::CloseRingsKwargs;
use crate::geoarrow::crs::{ExtensionMetadata, MetadataKwargs};
use crate::geoarrow::{coord, describe, ensure_same_crs_unless_bare, Geo, GeoDimension};
use geopolars_compute::inner_nulls::{
    multi_line_strings_with_null_parts_marked, multi_points_with_null_parts_marked,
    multi_polygons_with_null_parts_marked, propagate_inner_nulls_of_line_strings,
    propagate_inner_nulls_of_multi_line_strings, propagate_inner_nulls_of_multi_points,
    propagate_inner_nulls_of_multi_polygons, propagate_inner_nulls_of_points,
    propagate_inner_nulls_of_polygons, MaskedParts,
};
use geopolars_compute::native::{
    close_open_rings, multi_polygons_have_only_rings, polygons_have_only_rings,
    polygons_with_added_holes,
};
use geopolars_compute::with_const_flags;
use geopolars_core::{
    Kind, LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray, UncheckedStorage, ValidCoords, ValidList,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::Offsets;

/// The keyword arguments of the constructors that build polygons:
/// [`MetadataKwargs`], plus whether and how to close open rings.
/// These have to be mirrored on the Python side.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PolygonKwargs {
    /// The CRS the coordinates are in, in any form PROJ accepts.
    pub crs: Option<String>,
    /// How to close every open ring, as `close_rings` does.
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
    Ok(Geo::new(kind, dim, kwargs.apply(metadata, "parts")?))
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
            kwargs.apply(geo.metadata.clone(), "multipoint")?,
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
pub(super) fn validated_polygons(
    mut array: PolygonArray,
    close: Option<CloseRingsKwargs>,
) -> Box<dyn Array> {
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
    Ok(Geo::new(
        kind,
        dim,
        kwargs.apply(Default::default(), "coordinates")?,
    ))
}

/// The polygons or multipolygons `polygon_coords` and `multipolygon_coords` zip:
/// one ring per polygon, so nested one layer less deep than a `kind`.
fn zipped_exteriors(
    input_fields: &[Field],
    kwargs: &MetadataKwargs,
    kind: Kind,
) -> PolarsResult<Geo> {
    let dim = coordinates_of(input_fields, kind, kind.nesting() - 1)?;
    Ok(Geo::new(
        kind,
        dim,
        kwargs.apply(Default::default(), "coordinates")?,
    ))
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

pub(super) fn fields_of(inputs: &[Series]) -> Vec<Field> {
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

/// The single and the multi kind of the family `kind` is in.
pub(super) fn part_kinds(kind: Kind) -> (Kind, Kind) {
    match kind {
        Kind::Point | Kind::MultiPoint => (Kind::Point, Kind::MultiPoint),
        Kind::LineString | Kind::MultiLineString => (Kind::LineString, Kind::MultiLineString),
        Kind::Polygon | Kind::MultiPolygon => (Kind::Polygon, Kind::MultiPolygon),
    }
}

/// Whether a column of `found` can hold parts of kind `part`.
/// It can if `found` is `part` or its multi kind.
/// It can also be anything that holds the items of `part`
/// (the linestrings of a polygon, or the points of a linestring).
/// The lists around the column then decide which level it stands for.
fn holds_parts(found: Kind, part: Kind) -> bool {
    let (part, multi) = part_kinds(part);
    found == part
        || found == multi
        || match part {
            Kind::Polygon => holds_parts(found, Kind::LineString),
            Kind::LineString => holds_parts(found, Kind::Point),
            _ => false,
        }
}

/// How a column holds parts, as [parts_layout] finds it.
pub(super) struct PartsLayout {
    /// A list of parts per row, rather than one.
    pub(super) listed: bool,
    pub(super) dim: GeoDimension,
    /// `None` for bare coordinates.
    pub(super) metadata: Option<Arc<ExtensionMetadata>>,
}

/// Finds how a column of `dtype` holds parts of kind `part`.
/// Each part can be anything that [holds_parts], and bare coordinates can take the place of points.
/// Returns `None` for an untyped null.
pub(super) fn parts_layout(
    dtype: &DataType,
    part: Kind,
    name_in_err_msg: &str,
) -> PolarsResult<Option<PartsLayout>> {
    if matches!(dtype, DataType::Null) {
        return Ok(None);
    }
    let (part, multi) = part_kinds(part);
    let refused = || {
        polars_err!(
            SchemaMismatch: "expected the {name_in_err_msg} as one `{}` per row, a list of them, or a `{}` \
            ({}), got: {}",
            part.display(), multi.display(),
            match part {
                Kind::LineString => "a linestring can also be a list of its points or a `multipoint`, \
                    and a point its bare coordinates",
                Kind::Polygon => "a polygon can also be a list of its rings or a `multilinestring`, \
                    a ring a list of its points, a `linestring` or a `multipoint`, \
                    and a point its bare coordinates",
                _ => "a point can also be its bare coordinates",
            },
            dtype
        )
    };
    let (inner, lists) = coord::under_lists(dtype);
    let (nesting, dim, metadata) = match inner {
        DataType::Extension(..) => {
            let found = describe(inner).map_err(|_| refused())?;
            if !holds_parts(found.kind, part) {
                return Err(refused());
            }
            (
                found.kind.nesting(),
                found.dim,
                Some(found.metadata.clone()),
            )
        }
        bare => (0, coord::dimension_of(bare).ok_or_else(refused)?, None),
    };
    let listed = match (lists + nesting).checked_sub(part.nesting()) {
        Some(0) => false,
        Some(1) => true,
        _ => return Err(refused()),
    };
    Ok(Some(PartsLayout {
        listed,
        dim,
        metadata,
    }))
}

/// Checks that the column `parts` holds parts of kind `part` and dimension `dim` (see [parts_layout]).
/// Returns whether it holds a list of them per row, and its metadata.
/// The metadata is `None` for bare coordinates and an untyped null.
/// The CRS is left for the caller to check.
pub(super) fn listed_parts(
    parts: &Field,
    part: Kind,
    dim: GeoDimension,
    names_in_err_msg: (&str, &str),
) -> PolarsResult<(bool, Option<Arc<ExtensionMetadata>>)> {
    let (subject, name) = names_in_err_msg;
    let Some(layout) = parts_layout(parts.dtype(), part, name)? else {
        return Ok((true, None));
    };
    polars_ensure!(
        layout.dim == dim,
        SchemaMismatch: "the {subject} is {} but the {name} are {}", dim.tag(), layout.dim.tag()
    );
    Ok((layout.listed, layout.metadata))
}

/// [`PolygonKwargs`], plus what a null hole does.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SeparateRingsKwargs {
    /// The CRS the coordinates are in, in any form PROJ accepts.
    pub crs: Option<String>,
    /// How to close every open ring, as `close_rings` does.
    pub close: Option<CloseRingsKwargs>,
    null_parts: NullParts,
}

impl SeparateRingsKwargs {
    fn metadata(&self) -> MetadataKwargs {
        MetadataKwargs {
            crs: self.crs.clone(),
        }
    }
}

/// Checks the input fields of `polygon_rings_in_separate_columns`.
/// Returns the type of the result, and whether the holes are a list of rings per row.
/// The CRS in `kwargs` is set on an exterior or holes that declare none,
/// before their CRS are checked against each other.
fn separate_rings(input_fields: &[Field], kwargs: &MetadataKwargs) -> PolarsResult<(Geo, bool)> {
    let exterior = input_fields[0].dtype();
    let layout = parts_layout(exterior, Kind::LineString, "exterior")
        .ok()
        .flatten();
    let Some(layout) = layout.filter(|layout| !layout.listed) else {
        polars_bail!(
            SchemaMismatch: "expected the exterior as one ring per row: a `linestring`, \
            a `multipoint`, or a list of its points (or of their bare coordinates), got: {}",
            exterior
        );
    };
    let (listed, holes_metadata) = match input_fields.get(1) {
        Some(holes) => listed_parts(holes, Kind::LineString, layout.dim, ("exterior", "holes"))?,
        None => (false, None),
    };
    let exterior_metadata = (layout.metadata)
        .map(|metadata| kwargs.apply(metadata, "exterior"))
        .transpose()?;
    let holes_metadata = holes_metadata
        .map(|metadata| kwargs.apply(metadata, "holes"))
        .transpose()?;
    ensure_same_crs_unless_bare(
        exterior_metadata.as_deref(),
        holes_metadata.as_deref(),
        ("exterior", "holes"),
    )?;
    // A bare exterior takes the metadata of the holes.
    let metadata = match exterior_metadata.or(holes_metadata) {
        Some(metadata) => metadata,
        None => kwargs.apply(Default::default(), "exterior")?,
    };
    Ok((Geo::new(Kind::Polygon, layout.dim, metadata), listed))
}

fn kernel_output_type_polygon_rings_in_separate_columns(
    input_fields: &[Field],
    kwargs: SeparateRingsKwargs,
) -> PolarsResult<Field> {
    let (geo, _) = separate_rings(input_fields, &kwargs.metadata())?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

/// Rechunks `a` and `b`, repeating a single row to the length of the other.
pub(super) fn paired_rows(
    a: &Series,
    b: &Series,
    names_in_err_msg: (&str, &str),
) -> PolarsResult<(Series, Series)> {
    Ok(match (a.len(), b.len()) {
        (n, m) if n == m => (a.rechunk(), b.rechunk()),
        (n, 1) => (a.rechunk(), b.new_from_index(0, n).rechunk()),
        (1, m) => (a.new_from_index(0, m).rechunk(), b.rechunk()),
        (n, m) => {
            polars_bail!(ShapeMismatch: "the {} have {n} rows but the {} have {m}", names_in_err_msg.0, names_in_err_msg.1)
        }
    })
}

pub(super) fn storage_of(series: &Series) -> PolarsResult<Series> {
    Ok(match series.dtype() {
        DataType::Extension(..) => series.ext()?.storage().clone(),
        DataType::List(_) => series
            .list()?
            .apply_to_inner(&|inner| storage_of(&inner))?
            .into_series(),
        _ => series.clone(),
    })
}

pub(super) fn parts_storage(
    parts: &Series,
    multi: Kind,
    dim: GeoDimension,
) -> PolarsResult<Series> {
    match parts.dtype() {
        DataType::Null => {
            let storage = dim.storage(multi.nesting());
            Ok(Series::full_null(
                parts.name().clone(),
                parts.len(),
                &storage,
            ))
        }
        _ => storage_of(parts),
    }
}

fn one_per_row(len: usize, validity: Option<&Bitmap>) -> List {
    let lengths = std::iter::repeat_n(1, len);
    let offsets = Offsets::try_from_lengths(lengths).expect("one per row");
    List::from_offsets(offsets.into(), validity.cloned())
}

fn one_ring_per_row(exteriors: LineStringArray) -> PolygonArray {
    PolygonArray::from_parts(
        one_per_row(exteriors.rows.len(), exteriors.rows.validity()),
        exteriors.rows.ignoring_validity().clone(),
        exteriors.coords,
    )
}

pub(super) fn one_point_per_row(points: PointArray) -> MultiPointArray {
    MultiPointArray::from_parts(
        one_per_row(points.coords.len(), points.coords.validity()),
        points.coords.ignoring_validity().clone(),
    )
}

pub(super) fn one_line_per_row(lines: LineStringArray) -> MultiLineStringArray {
    MultiLineStringArray::from_parts(
        one_per_row(lines.rows.len(), lines.rows.validity()),
        lines.rows.ignoring_validity().clone(),
        lines.coords,
    )
}

pub(super) fn one_polygon_per_row(polygons: PolygonArray) -> MultiPolygonArray {
    MultiPolygonArray::from_parts(
        one_per_row(polygons.rows.len(), polygons.rows.validity()),
        polygons.rows.ignoring_validity().clone(),
        polygons.rings,
        polygons.coords,
    )
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum NullParts {
    /// The null part is left out.
    Drop,
    /// The null part makes the result null.
    #[default]
    Null,
}

/// Combines the validity of `base` and `parts` into that of a result built out of them.
/// A null row of `parts` only makes the result null with [NullParts::Null].
pub(super) fn rows_validity(base: &List, parts: &List, null_parts: NullParts) -> Option<Bitmap> {
    match null_parts {
        NullParts::Drop => base.validity().cloned(),
        NullParts::Null => combine_validities_and(base.validity(), parts.validity()),
    }
}

/// Reads a chunk of linestrings as multilinestrings.
/// `chunk` only holds a single linestring per row if `listed=false`.
/// A linestring holding a null value makes its row null.
/// In a list of linestrings, [NullParts::Drop] unsets it in the mask instead.
pub(super) fn lines_per_row(
    chunk: &dyn Array,
    listed: bool,
    null_parts: NullParts,
) -> PolarsResult<MaskedParts<MultiLineStringArray>> {
    let storage = UncheckedStorage::new(chunk)?;
    Ok(match (listed, null_parts) {
        (false, _) => one_line_per_row(propagate_inner_nulls_of_line_strings(storage)).into(),
        (true, NullParts::Drop) => multi_line_strings_with_null_parts_marked(storage),
        (true, NullParts::Null) => propagate_inner_nulls_of_multi_line_strings(storage).into(),
    })
}

/// [lines_per_row], for points.
pub(super) fn points_per_row(
    chunk: &dyn Array,
    listed: bool,
    null_parts: NullParts,
) -> PolarsResult<MaskedParts<MultiPointArray>> {
    let storage = UncheckedStorage::new(chunk)?;
    Ok(match (listed, null_parts) {
        (false, _) => one_point_per_row(propagate_inner_nulls_of_points(storage)).into(),
        (true, NullParts::Drop) => multi_points_with_null_parts_marked(storage),
        (true, NullParts::Null) => propagate_inner_nulls_of_multi_points(storage).into(),
    })
}

/// [lines_per_row], for polygons.
pub(super) fn polygons_per_row(
    chunk: &dyn Array,
    listed: bool,
    null_parts: NullParts,
) -> PolarsResult<MaskedParts<MultiPolygonArray>> {
    let storage = UncheckedStorage::new(chunk)?;
    Ok(match (listed, null_parts) {
        (false, _) => one_polygon_per_row(propagate_inner_nulls_of_polygons(storage)).into(),
        (true, NullParts::Drop) => multi_polygons_with_null_parts_marked(storage),
        (true, NullParts::Null) => propagate_inner_nulls_of_multi_polygons(storage).into(),
    })
}

/// Assembles one chunk of polygons from one chunk of exterior rings and one of holes.
/// The holes only hold a single ring per row if `listed=false`.
/// See `polygons_from_rings_in_separate_columns` for the null values.
fn assembled_polygons(
    exteriors: &dyn Array,
    holes: Option<(&dyn Array, bool, NullParts)>,
    close: Option<CloseRingsKwargs>,
) -> PolarsResult<Box<dyn Array>> {
    let exteriors = one_ring_per_row(propagate_inner_nulls_of_line_strings(
        UncheckedStorage::new(exteriors)?,
    ));
    let Some((holes, listed, null_parts)) = holes else {
        // FIXME: this copies every coordinate into new buffers for nothing.
        // `exteriors` already is a trimmed polygon of one ring per row, without a copy,
        // so this can be `return Ok(validated_polygons(exteriors, close))`.
        // `polygons_with_added_holes` can then take `holes` without the `Option`.
        // This affects `polygons_from_coordinates`,
        // and `polygons_from_rings_in_separate_columns` without holes.
        let validity = exteriors.rows.validity().cloned();
        let polygons = polygons_with_added_holes(&exteriors, None, validity);
        return Ok(validated_polygons(polygons, close));
    };

    let holes = lines_per_row(holes, listed, null_parts)?;
    let validity = rows_validity(&exteriors.rows, holes.rows(), null_parts);
    let polygons = polygons_with_added_holes(&exteriors, Some(&holes), validity);
    Ok(validated_polygons(polygons, close))
}

/// See `polygons_from_rings_in_separate_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon_rings_in_separate_columns)]
fn polygon_rings_in_separate_columns(
    inputs: &[Series],
    kwargs: SeparateRingsKwargs,
) -> PolarsResult<Series> {
    let (geo, listed) = separate_rings(&fields_of(inputs), &kwargs.metadata())?;
    let exteriors = storage_of(&inputs[0])?;
    let chunk = match inputs.get(1) {
        Some(holes) => {
            let holes = parts_storage(holes, Kind::MultiLineString, geo.dim())?;
            let (exteriors, holes) = paired_rows(&exteriors, &holes, ("exteriors", "holes"))?;
            let holes = (holes.chunks()[0].as_ref(), listed, kwargs.null_parts);
            assembled_polygons(exteriors.chunks()[0].as_ref(), Some(holes), kwargs.close)?
        }
        None => assembled_polygons(exteriors.rechunk().chunks()[0].as_ref(), None, kwargs.close)?,
    };
    // SAFETY: the chunk is the storage of a polygon of `geo`'s dimension.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), vec![chunk], &geo.dtype())
    })
}
