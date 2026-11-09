//! Lining up the storage chunks of two geometry columns for a kernel,
//! row by row or with a single geometry on either side against every row of the other.

use polars::prelude::*;
use polars_arrow::compute::utils::combine_validities_and;
use polars_core::utils::align_chunks_binary;

use super::storage_as_single_chunk;

/// The storage chunks of two geometry columns, lined up for a kernel.
pub enum GeometryPairs {
    /// Chunks of the same length, to pair row by row.
    Pairwise(Vec<(ArrayRef, ArrayRef)>),
    /// A single geometry, which is not null, against every chunk of the other column.
    AgainstOne {
        one: ArrayRef,
        many: Vec<ArrayRef>,
        one_is_a: bool,
    },
    /// A null single geometry: the length of each all-null output chunk.
    AllNull(Vec<usize>),
}

/// Lines up `a` and `b`, splitting columns of the same length at the same rows.
/// This only copies when both have several chunks that do not line up.
/// Both must be stored alike: both points, or both lists.
pub fn aligned_pairs(a: &Series, b: &Series) -> PolarsResult<GeometryPairs> {
    if a.len() != b.len() {
        return against_one_pairs(a, b);
    }
    let (a, b) = (a.ext()?.storage(), b.ext()?.storage());
    let zipped =
        |a: &[ArrayRef], b: &[ArrayRef]| a.iter().cloned().zip(b.iter().cloned()).collect();
    Ok(GeometryPairs::Pairwise(match a.dtype() {
        DataType::Struct(_) => {
            let (a, b) = align_chunks_binary(a.struct_()?, b.struct_()?);
            zipped(a.chunks(), b.chunks())
        }
        _ => {
            let (a, b) = align_chunks_binary(a.list()?, b.list()?);
            zipped(a.chunks(), b.chunks())
        }
    }))
}

/// Lines up `a` and `b`, rechunking columns of the same length into a single chunk each,
/// so they can be of any two kinds.
/// A row that is null on one side is made null on the other too.
/// This is for a kernel that converts every row to GEOS up front,
/// where a row paired with a null could otherwise still fail.
pub fn single_chunk_masked_pairs(a: &Series, b: &Series) -> PolarsResult<GeometryPairs> {
    if a.len() != b.len() {
        return against_one_pairs(a, b);
    }
    let (a, b) = (storage_as_single_chunk(a)?, storage_as_single_chunk(b)?);
    let validity = combine_validities_and(a.validity(), b.validity());
    Ok(GeometryPairs::Pairwise(vec![(
        a.with_validity(validity.clone()),
        b.with_validity(validity),
    )]))
}

/// For two columns of different lengths, one of which must be a single geometry.
fn against_one_pairs(a: &Series, b: &Series) -> PolarsResult<GeometryPairs> {
    let (one, many, one_is_a) = match (a.len(), b.len()) {
        (1, _) => (a, b, true),
        (_, 1) => (b, a, false),
        (n, m) => polars_bail!(ShapeMismatch: "cannot pair {n} geometries with {m}"),
    };
    let one = storage_as_single_chunk(one)?;
    let many = many.ext()?.storage().chunks();
    if one.is_null(0) {
        return Ok(GeometryPairs::AllNull(
            many.iter().map(|many| many.len()).collect(),
        ));
    }
    Ok(GeometryPairs::AgainstOne {
        one,
        many: many.clone(),
        one_is_a,
    })
}
