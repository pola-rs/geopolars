//! Glue for the GeoArrow data types, and dispatching to it.
//!
//! Expressions never name a concrete geometry:
//! They ask [`describe`] what a column holds and dispatch on the [`Kind`] and [`Dimension`] it reports,
//! so adding a geometry does not touch `crate::expr`.

pub mod bbox;
pub mod coord;
pub mod crs;
pub mod decode;
pub mod encoded;
mod geo;
mod kind;

pub use geo::Geo;
pub use geopolars_core::GeoDimension;
pub use kind::Kind;

use std::any::Any;
use std::sync::Arc;

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_core::datatypes::extension::{register_extension_type, ExtensionTypeInstance};

use geopolars_core::{
    downcast, Coords, Geometry, GeometryArray, LineStringArray, MultiLineStringArray,
    MultiPointArray, MultiPolygonArray, PointArray, PolygonArray,
};

use bbox::BoxFactory;
use crs::ExtensionMetadata;
use encoded::{EncodedFactory, Encoding};
use geo::GeoFactory;

/// Populate this library's extension-type registry.
/// This exposes the types from this plugin's library.
/// This cannot run lazily.
///
/// Registering also needs to happen in the host `polars` wheel,
/// which is done with `pl.register_extension_type`.
/// Both are needed, and they must agree on names.
pub fn register() -> PolarsResult<()> {
    for kind in Kind::ALL {
        register_extension_type(kind.name(), Some(Arc::new(GeoFactory(kind))))?;
    }
    for encoding in Encoding::ALL {
        register_extension_type(encoding.name(), Some(Arc::new(EncodedFactory(encoding))))?;
    }
    register_extension_type(bbox::NAME, Some(Arc::new(BoxFactory)))?;
    Ok(())
}

/// A view of a geometry column, containing only what an expression over it needs.
pub struct GeoColumn<'a> {
    /// The [`ExtensionTypeInstance`] the column came in, so the result can be
    /// put back under it without losing metadata.
    pub typ: &'a ExtensionTypeInstance,
    pub kind: Kind,
    pub dim: GeoDimension,
    pub metadata: &'a Arc<ExtensionMetadata>,
}

/// Parses a geometry column's dtype (as a polars [DataType]) into a [GeoColumn].
pub fn describe(dtype: &DataType) -> PolarsResult<GeoColumn<'_>> {
    let DataType::Extension(typ, storage) = dtype else {
        polars_bail!(SchemaMismatch: "expected a {} column, got: {}", Kind::names(), dtype);
    };
    let Some(kind) = Kind::from_name(&typ.name()) else {
        polars_bail!(SchemaMismatch: "expected a {} column, got: {}", Kind::names(), dtype);
    };
    // `GeoFactory` already parsed everything, unless it had to give up.
    if let Some(geo) = (typ.0.as_ref() as &dyn Any).downcast_ref::<Geo>() {
        return Ok(GeoColumn {
            typ,
            kind: geo.kind(),
            dim: geo.dim(),
            metadata: geo.metadata(),
        });
    }

    polars_ensure!(
        coord::dimension_of_storage(storage, kind.nesting()).is_some(),
        // An interleaved encoding, most likely.
        SchemaMismatch: "expected a `{}` column with separated \
        x/y[/z][/m] coordinates, got: {}", kind.name(), dtype
    );
    ExtensionMetadata::parse(typ.serialize_metadata().as_deref())?;
    polars_bail!(
        SchemaMismatch: "`{}` is not registered with this library, got: {}", kind.name(), dtype
    )
}

/// The [Kind] of a geometry column, see [describe].
pub fn kind_of(geometries: &Series) -> PolarsResult<Kind> {
    Ok(describe(geometries.dtype())?.kind)
}

/// The storage of a geometry column, rechunked into a single chunk.
pub fn storage_as_single_chunk(geometries: &Series) -> PolarsResult<ArrayRef> {
    Ok(geometries.ext()?.storage().rechunk().chunks()[0].clone())
}

/// Wraps a chunk of a `kind` column's storage, for kernels that take any kind of geometry.
pub fn geometry_array(chunk: &dyn Array, kind: Kind) -> PolarsResult<GeometryArray> {
    let rows = |chunk| downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => GeometryArray::Point(PointArray::new(Coords::new(chunk)?)),
        Kind::MultiPoint => GeometryArray::MultiPoint(MultiPointArray::new(rows(chunk)?)?),
        Kind::LineString => GeometryArray::LineString(LineStringArray::new(rows(chunk)?)?),
        Kind::MultiLineString => {
            GeometryArray::MultiLineString(MultiLineStringArray::new(rows(chunk)?)?)
        }
        Kind::Polygon => GeometryArray::Polygon(PolygonArray::new(rows(chunk)?)?),
        Kind::MultiPolygon => GeometryArray::MultiPolygon(MultiPolygonArray::new(rows(chunk)?)?),
    })
}

/// Runs a kernel on two geometry columns of any kinds, row by row,
/// or with a single geometry on either side against every row of the other.
/// `against_one` gets that single geometry first, with `one_is_a=true` if it came from `a`.
/// A row that is null on one side is made null on the other too,
/// so that the kernel never sends it to GEOS, where it could fail.
pub fn map_geometry_pairs<T>(
    a: &Series,
    b: &Series,
    pairwise: impl Fn(&GeometryArray, &GeometryArray) -> PolarsResult<T>,
    against_one: impl Fn(&GeometryArray, &GeometryArray, bool) -> PolarsResult<T>,
) -> PolarsResult<Vec<T>> {
    match (a.len(), b.len()) {
        // The kinds can differ, but `aligned_chunks` needs two storages of one physical type.
        // So both sides are made a single chunk instead.
        (n, m) if n == m => {
            let (a_chunk, b_chunk) = (storage_as_single_chunk(a)?, storage_as_single_chunk(b)?);
            let validity = combine_validities_and(a_chunk.validity(), b_chunk.validity());
            let (a_rows, b_rows) = (
                geometry_array(
                    a_chunk.with_validity(validity.clone()).as_ref(),
                    kind_of(a)?,
                )?,
                geometry_array(b_chunk.with_validity(validity).as_ref(), kind_of(b)?)?,
            );
            Ok(vec![pairwise(&a_rows, &b_rows)?])
        }
        (1, _) => map_geometry_pairs_against_one(a, b, true, against_one),
        (_, 1) => map_geometry_pairs_against_one(b, a, false, against_one),
        (n, m) => polars_bail!(ShapeMismatch: "cannot pair {n} geometries with {m}"),
    }
}

fn map_geometry_pairs_against_one<T>(
    one: &Series,
    many: &Series,
    one_is_a: bool,
    against_one: impl Fn(&GeometryArray, &GeometryArray, bool) -> PolarsResult<T>,
) -> PolarsResult<Vec<T>> {
    let one_chunk = storage_as_single_chunk(one)?;
    let one_is_null = one_chunk.is_null(0);
    let one = geometry_array(one_chunk.as_ref(), kind_of(one)?)?;
    let kind = kind_of(many)?;
    many.ext()?
        .storage()
        .chunks()
        .iter()
        .map(|many| {
            // A null `one` makes every row null.
            let many = if one_is_null {
                many.with_validity(Some(Bitmap::new_zeroed(many.len())))
            } else {
                many.clone()
            };
            against_one(&one, &geometry_array(many.as_ref(), kind)?, one_is_a)
        })
        .collect()
}

/// Errors unless `a` and `b` declare the same CRS, or neither declares one.
/// Should be used as a backup, as this is runtime, not plan time.
pub fn ensure_same_crs(a: &GeoColumn, b: &GeoColumn) -> PolarsResult<()> {
    ensure_same_crs_unless_bare(
        Some(a.metadata),
        Some(b.metadata),
        ("first geometry", "second geometry"),
    )
}

/// [ensure_same_crs], but `None` (bare coordinates, or an untyped null) goes with any CRS.
pub fn ensure_same_crs_unless_bare(
    a: Option<&ExtensionMetadata>,
    b: Option<&ExtensionMetadata>,
    names_in_err_msg: (&str, &str),
) -> PolarsResult<()> {
    let (Some(a), Some(b)) = (a, b) else {
        return Ok(());
    };
    let (name_a, name_b) = names_in_err_msg;
    match (a.declares_crs(), b.declares_crs()) {
        (false, false) => Ok(()),
        (true, true) => {
            let (crs_a, crs_b) = (a.crs()?, b.crs()?);
            polars_ensure!(
                crs_a == crs_b,
                SchemaMismatch: "the {name_a} and the {name_b} declare different CRSs, {crs_a} and {crs_b}; \
                use `to_crs` to bring one onto the other first"
            );
            Ok(())
        }
        (true, false) | (false, true) => polars_bail!(
            SchemaMismatch: "only one of the {name_a} and the {name_b} declares a CRS; \
            declare the same one on both"
        ),
    }
}

/// Calls `f` with `row(i)` for every row `i` of `chunk`, or `None` where it is null.
fn rows<'a>(
    chunk: &dyn Array,
    f: &mut impl FnMut(Option<Geometry<'a>>) -> PolarsResult<()>,
    row: impl Fn(usize) -> Geometry<'a>,
) -> PolarsResult<()> {
    (0..chunk.len()).try_for_each(|i| f((!chunk.is_null(i)).then(|| row(i))))
}

/// Calls `f` with every row of `chunk`, a chunk of a `kind` column's storage:
/// the geometry, or `None` for a null.
/// For the `wkb` and `wkt` writers, which take geometries through geo-traits.
pub fn for_each_geometry(
    chunk: &dyn Array,
    kind: Kind,
    mut f: impl FnMut(Option<Geometry<'_>>) -> PolarsResult<()>,
) -> PolarsResult<()> {
    match kind {
        Kind::Point => {
            let array = PointArray::new(Coords::new(chunk)?);
            rows(chunk, &mut f, |i| Geometry::Point(array.get(i)))
        }
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            rows(chunk, &mut f, |i| Geometry::LineString(array.get(i)))
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            rows(chunk, &mut f, |i| Geometry::MultiPoint(array.get(i)))
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            rows(chunk, &mut f, |i| Geometry::Polygon(array.get(i)))
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            rows(chunk, &mut f, |i| Geometry::MultiLineString(array.get(i)))
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            rows(chunk, &mut f, |i| Geometry::MultiPolygon(array.get(i)))
        }
    }
}
