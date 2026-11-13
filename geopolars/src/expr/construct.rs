//! Building a geometry out of the parts it is made of.

use std::sync::Arc;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::close_rings::CloseRingsKwargs;
use crate::geoarrow::crs::{ExtensionMetadata, MetadataKwargs};
use crate::geoarrow::{coord, describe, ensure_same_crs_unless_bare, geometry_series, Geo};
use geopolars_core::{
    assembled_multi_polygons, assembled_polygons, validated_storage, NestedKind, NullParts,
};
use geopolars_views::{GeoDimension, Kind};

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

/// Runs [validated_storage] on every chunk of `storage`, and gives the result as a `geo` column.
fn validated(
    storage: &Series,
    geo: Geo,
    kind: NestedKind,
    close: Option<CloseRingsKwargs>,
) -> PolarsResult<Series> {
    let close = close.map(Into::into);
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| validated_storage(chunk.as_ref(), kind, close))
        .collect::<PolarsResult<Vec<_>>>()?;
    geometry_series(storage.name().clone(), geo, chunks)
}

/// Gather a list of `part`s into one geometry of `kind` per row.
/// `close` says how to close open rings, and only means something for kinds that have them.
fn gather(
    parts: &Series,
    kwargs: &MetadataKwargs,
    close: Option<CloseRingsKwargs>,
    kind: NestedKind,
    part: Kind,
) -> PolarsResult<Series> {
    let geo = gathered(parts.dtype(), kwargs, kind.kind(), part)?;
    // This only changes dtype, nothing on the data.
    let storage = parts.list()?.apply_to_inner(&|part| {
        Ok(match part.dtype() {
            DataType::Extension(_, _) => part.ext()?.storage().clone(),
            _ => part,
        })
    })?;

    validated(&storage.into_series(), geo, kind, close)
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
        _ => gather(vertices, &kwargs, None, NestedKind::LineString, Kind::Point),
    }
}

/// See `polygons_from_rings_in_single_column`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon)]
fn polygon(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    gather(
        &inputs[0],
        &kwargs.metadata(),
        kwargs.close,
        NestedKind::Polygon,
        Kind::LineString,
    )
}

/// See `multipoint_from_points`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipoint)]
fn multipoint(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    // Note that this is basically the same as linestring, but just a different [Kind].
    gather(
        &inputs[0],
        &kwargs,
        None,
        NestedKind::MultiPoint,
        Kind::Point,
    )
}

/// See `multilinestring_from_linestrings`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multilinestring)]
fn multilinestring(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    gather(
        &inputs[0],
        &kwargs,
        None,
        NestedKind::MultiLineString,
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
        NestedKind::MultiPolygon,
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
    kind: NestedKind,
) -> PolarsResult<Series> {
    let geo = zipped(&fields_of(inputs), kwargs, kind.kind())?;
    let storage = interleaved_columns(inputs, kind.kind().nesting())?;
    validated(&storage, geo, kind, close)
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
    zip(inputs, &kwargs, None, NestedKind::LineString)
}

/// See `polygons_from_coordinates`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon_coords)]
fn polygon_coords(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    let geo = zipped_exteriors(&fields_of(inputs), &kwargs.metadata(), Kind::Polygon)?;
    let exteriors = interleaved_columns(inputs, Kind::LineString.nesting())?;
    let chunks = exteriors
        .chunks()
        .iter()
        .map(|chunk| assembled_polygons(chunk.as_ref(), None, kwargs.close.map(Into::into)))
        .collect::<PolarsResult<Vec<_>>>()?;
    geometry_series(exteriors.name().clone(), geo, chunks)
}

/// See `polygons_from_nested_coordinates`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon_nested_coords)]
fn polygon_nested_coords(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    zip(
        inputs,
        &kwargs.metadata(),
        kwargs.close,
        NestedKind::Polygon,
    )
}

/// See `multipoint_from_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipoint_coords)]
fn multipoint_coords(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs, None, NestedKind::MultiPoint)
}

/// See `multilinestring_from_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multilinestring_coords)]
fn multilinestring_coords(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs, None, NestedKind::MultiLineString)
}

/// See `multi_polygons_from_coordinates`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipolygon_coords)]
fn multipolygon_coords(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    let geo = zipped_exteriors(&fields_of(inputs), &kwargs.metadata(), Kind::MultiPolygon)?;
    let exteriors = interleaved_columns(inputs, Kind::MultiLineString.nesting())?;
    let chunks = exteriors
        .chunks()
        .iter()
        .map(|chunk| assembled_multi_polygons(chunk.as_ref(), kwargs.close.map(Into::into)))
        .collect::<PolarsResult<Vec<_>>>()?;
    geometry_series(exteriors.name().clone(), geo, chunks)
}

/// See `multi_polygons_from_nested_coordinates`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipolygon_nested_coords)]
fn multipolygon_nested_coords(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    zip(
        inputs,
        &kwargs.metadata(),
        kwargs.close,
        NestedKind::MultiPolygon,
    )
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
    #[serde(with = "NullPartsDef")]
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

#[derive(Deserialize)]
#[serde(remote = "NullParts", rename_all = "snake_case")]
pub(super) enum NullPartsDef {
    Drop,
    Null,
}

/// See `polygons_from_rings_in_separate_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon_rings_in_separate_columns)]
fn polygon_rings_in_separate_columns(
    inputs: &[Series],
    kwargs: SeparateRingsKwargs,
) -> PolarsResult<Series> {
    let (geo, listed) = separate_rings(&fields_of(inputs), &kwargs.metadata())?;
    let exteriors = storage_of(&inputs[0])?;
    let close = kwargs.close.map(Into::into);
    let chunk = match inputs.get(1) {
        Some(holes) => {
            let holes = parts_storage(holes, Kind::MultiLineString, geo.dim())?;
            let (exteriors, holes) = paired_rows(&exteriors, &holes, ("exteriors", "holes"))?;
            let holes = (holes.chunks()[0].as_ref(), listed, kwargs.null_parts);
            assembled_polygons(exteriors.chunks()[0].as_ref(), Some(holes), close)?
        }
        None => assembled_polygons(exteriors.rechunk().chunks()[0].as_ref(), None, close)?,
    };
    geometry_series(inputs[0].name().clone(), geo, vec![chunk])
}
