//! Whether two geometries are equal.
//! `is_equal_identical` compares the coordinates natively
//! (see [`geopolars_compute::native::points_are_close`] and the kernels of the other kinds).
//! `is_equal_topological` forwards to GEOS
//! (see [`geopolars_compute::geos::topologically_equal_pairwise`]).

use geopolars_compute::geos::{
    topologically_equal_against_one, topologically_equal_pairwise, GeosGeometries, ToGeosRows,
};
use geopolars_compute::native::{
    line_strings_are_close, multi_line_strings_are_close, multi_points_are_close,
    multi_polygons_are_close, points_are_close, polygons_are_close, Coincidence,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_core::utils::align_chunks_binary;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{describe, ensure_same_crs, Kind};

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsEqualIdenticalKwargs {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    nans_equal: bool,
    unique_z: bool,
    unique_m: bool,
}

/// For [`is_equal_identical`]: a `bool`, named after the first geometry.
///
/// Both have to be the same kind and dimension, in the same CRS (or neither declaring one).
fn kernel_output_type_is_equal_identical(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "is_equal takes two geometries, got {} inputs", input_fields.len());
    };
    let (geo_a, geo_b) = (describe(a.dtype())?, describe(b.dtype())?);
    polars_ensure!(
        geo_a.kind == geo_b.kind && geo_a.dim == geo_b.dim,
        SchemaMismatch: "is_equal with mode=\"identical\" expects two geometries of the same type, got: {} and {}",
        a.dtype(), b.dtype()
    );
    ensure_same_crs(&geo_a, &geo_b)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

/// The kernel of `kind` on one chunk of `a`'s storage and one of `b`'s.
/// `b` is as long as `a`, or a single row.
fn chunk_is_close(
    a: &dyn Array,
    b: &dyn Array,
    kind: Kind,
    kwargs: &IsEqualIdenticalKwargs,
) -> PolarsResult<BooleanArray> {
    let coincidence = Coincidence::new(kwargs.abs_tol, kwargs.rel_tol);
    let (unique_z, unique_m, nans_equal) = (kwargs.unique_z, kwargs.unique_m, kwargs.nans_equal);
    let rows = |chunk| downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_are_close(
            &PointArray::new(Coords::new(a)?),
            &PointArray::new(Coords::new(b)?),
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiPoint => multi_points_are_close(
            &MultiPointArray::new_untrimmed(rows(a)?)?,
            &MultiPointArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::LineString => line_strings_are_close(
            &LineStringArray::new_untrimmed(rows(a)?)?,
            &LineStringArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiLineString => multi_line_strings_are_close(
            &MultiLineStringArray::new_untrimmed(rows(a)?)?,
            &MultiLineStringArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::Polygon => polygons_are_close(
            &PolygonArray::new_untrimmed(rows(a)?)?,
            &PolygonArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
        Kind::MultiPolygon => multi_polygons_are_close(
            &MultiPolygonArray::new_untrimmed(rows(a)?)?,
            &MultiPolygonArray::new_untrimmed(rows(b)?)?,
            coincidence,
            unique_z,
            unique_m,
            nans_equal,
        ),
    })
}

/// The chunks of the storages of `a` and `b`, which are the same length, split at the same rows.
fn aligned_chunks(a: &Series, b: &Series) -> PolarsResult<Vec<(ArrayRef, ArrayRef)>> {
    let (a, b) = (a.ext()?.storage(), b.ext()?.storage());
    let zipped =
        |a: &[ArrayRef], b: &[ArrayRef]| a.iter().cloned().zip(b.iter().cloned()).collect();
    Ok(match a.dtype() {
        DataType::Struct(_) => {
            let (a, b) = align_chunks_binary(a.struct_()?, b.struct_()?);
            zipped(a.chunks(), b.chunks())
        }
        _ => {
            let (a, b) = align_chunks_binary(a.list()?, b.list()?);
            zipped(a.chunks(), b.chunks())
        }
    })
}

/// [chunk_is_close] for every chunk of `many`, against the single row of `one`.
/// Closeness goes both ways, so this serves a single geometry on either side.
fn every_is_close_to_one(
    one: &Series,
    many: &Series,
    kind: Kind,
    kwargs: &IsEqualIdenticalKwargs,
) -> PolarsResult<Vec<BooleanArray>> {
    let one = one.ext()?.storage().rechunk();
    let one = one.chunks()[0].as_ref();
    many.ext()?
        .storage()
        .chunks()
        .iter()
        .map(|many| chunk_is_close(many.as_ref(), one, kind, kwargs))
        .collect()
}

/// See `is_equal`, with `mode="identical"`.
#[polars_expr(output_type_func=kernel_output_type_is_equal_identical)]
fn is_equal_identical(inputs: &[Series], kwargs: IsEqualIdenticalKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let kind = describe(a.dtype())?.kind;
    let chunks = match (a.len(), b.len()) {
        (n, m) if n == m => aligned_chunks(a, b)?
            .iter()
            .map(|(a, b)| chunk_is_close(a.as_ref(), b.as_ref(), kind, &kwargs))
            .collect::<PolarsResult<Vec<_>>>()?,
        (1, _) => every_is_close_to_one(a, b, kind, &kwargs)?,
        (_, 1) => every_is_close_to_one(b, a, kind, &kwargs)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot compare {n} geometries against {m}"),
    };
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}

/// See python
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsEqualTopologicalKwargs {
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// For [`is_equal_topological`]: a `bool`, named after the first geometry.
///
/// Both can be of any kind and dimension, in the same CRS (or neither declaring one).
fn kernel_output_type_is_equal_topological(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "is_equal takes two geometries, got {} inputs", input_fields.len());
    };
    ensure_same_crs(&describe(a.dtype())?, &describe(b.dtype())?)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

/// One chunk of the storage of a `kind`, converted for GEOS.
fn chunk_to_geos_rows(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &IsEqualTopologicalKwargs,
) -> PolarsResult<GeosGeometries> {
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let rows = |chunk| downcast::<ListArray<i64>>(chunk, "a list of parts");
    match kind {
        Kind::Point => PointArray::new(Coords::new(chunk)?).to_geos_rows(ignore, early),
        Kind::MultiPoint => MultiPointArray::new(rows(chunk)?)?.to_geos_rows(ignore, early),
        Kind::LineString => LineStringArray::new(rows(chunk)?)?.to_geos_rows(ignore, early),
        Kind::MultiLineString => {
            MultiLineStringArray::new(rows(chunk)?)?.to_geos_rows(ignore, early)
        }
        Kind::Polygon => PolygonArray::new(rows(chunk)?)?.to_geos_rows(ignore, early),
        Kind::MultiPolygon => MultiPolygonArray::new(rows(chunk)?)?.to_geos_rows(ignore, early),
    }
    .into_polars()
}

/// `geometries` as a single chunk, converted for GEOS.
fn series_to_geos_rows(
    geometries: &Series,
    kwargs: &IsEqualTopologicalKwargs,
) -> PolarsResult<GeosGeometries> {
    let kind = describe(geometries.dtype())?.kind;
    let storage = geometries.ext()?.storage().rechunk();
    chunk_to_geos_rows(storage.chunks()[0].as_ref(), kind, kwargs)
}

/// [topologically_equal_against_one] for every chunk of `many`, against the single row of `one`.
fn every_topologically_equal_to_one(
    one: &Series,
    many: &Series,
    kwargs: &IsEqualTopologicalKwargs,
) -> PolarsResult<Vec<BooleanArray>> {
    let one = series_to_geos_rows(one, kwargs)?;
    let one = &one.geometries[0];
    let kind = describe(many.dtype())?.kind;
    many.ext()?
        .storage()
        .chunks()
        .iter()
        .map(|many| {
            let many = chunk_to_geos_rows(many.as_ref(), kind, kwargs)?;
            topologically_equal_against_one(one, &many, kwargs.ignore_per_row_errors)
                .into_polars()
        })
        .collect()
}

/// See `is_equal`, with `mode="topological"`.
#[polars_expr(output_type_func=kernel_output_type_is_equal_topological)]
fn is_equal_topological(
    inputs: &[Series],
    kwargs: IsEqualTopologicalKwargs,
) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let chunks = match (a.len(), b.len()) {
        // `aligned_chunks` takes two storages of one physical type, and the kinds can differ,
        // so both sides are made a single chunk instead, as in `rings_overlap`.
        (n, m) if n == m => {
            let (a, b) = (series_to_geos_rows(a, &kwargs)?, series_to_geos_rows(b, &kwargs)?);
            vec![topologically_equal_pairwise(&a, &b, kwargs.ignore_per_row_errors).into_polars()?]
        }
        (1, _) => every_topologically_equal_to_one(a, b, &kwargs)?,
        (_, 1) => every_topologically_equal_to_one(b, a, &kwargs)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot compare {n} geometries against {m}"),
    };
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
