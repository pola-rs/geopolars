//! How long a linestring is, as a [LengthMetric] measures it.
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! Every function here is generic over `Z`.
//! If `Z=true`, the difference in z of a segment counts towards its length.
//!
//! ## `IGNORE_NAN`
//!
//! Every function here is generic over `IGNORE_NAN`.
//! If `IGNORE_NAN=true`, a vertex with a NaN is skipped, so the line runs on to the next one.

use geopolars_views::{LineStringArray, MultiLineStringArray, ScalarLineString};
use polars_arrow::array::PrimitiveArray;

use super::DistanceMetric;
use crate::per_row::per_row;
use crate::{GeodesicMetric, Planar};

/// A [`DistanceMetric`] that can also measure how long a linestring is, segment by segment.
pub trait LengthMetric: DistanceMetric {}

impl LengthMetric for Planar {}

impl LengthMetric for GeodesicMetric {}

/// The length of `line`: fewer than two vertices have a length of `0.0`.
#[inline(always)]
fn line_length<const Z: bool, const IGNORE_NAN: bool, M: LengthMetric>(
    metric: &M,
    line: &ScalarLineString,
) -> Result<f64, M::Error> {
    let mut vertices = line
        .positions::<Z>()
        .filter(|&[x, y, z]| !IGNORE_NAN || !(x.is_nan() || y.is_nan() || (Z && z.is_nan())));
    let Some(first) = vertices.next() else {
        return Ok(0.0);
    };
    let mut previous = metric.vertex(first)?;
    let mut total = 0.0;
    for position in vertices {
        let here = metric.vertex(position)?;
        total += metric.segment::<Z>(previous, here);
        previous = here;
    }
    Ok(total)
}

/// The length of every linestring of a chunk.
pub fn line_string_lengths<const Z: bool, const IGNORE_NAN: bool, M: LengthMetric>(
    metric: &M,
    array: &LineStringArray,
) -> Result<PrimitiveArray<f64>, M::Error> {
    per_row(&array.rows, |i| {
        line_length::<Z, IGNORE_NAN, _>(metric, &array.get(i))
    })
}

/// The length of every multilinestring of a chunk, as the sum of its parts.
pub fn multi_line_string_lengths<const Z: bool, const IGNORE_NAN: bool, M: LengthMetric>(
    metric: &M,
    array: &MultiLineStringArray,
) -> Result<PrimitiveArray<f64>, M::Error> {
    per_row(&array.rows, |i| {
        array
            .get(i)
            .line_strings()
            .map(|line| line_length::<Z, IGNORE_NAN, _>(metric, &line))
            .sum()
    })
}
