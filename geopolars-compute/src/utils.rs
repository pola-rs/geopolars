//! Helpers shared by kernels.
//! See "Helpers" in docs/source/development/geo/kernel-implementation-guide.md.

use std::ops::Range;

use geopolars_views::output::succeeded;
use geopolars_views::{List, ValidCoords};
use polars_arrow::compute::utils::combine_validities_and;

/// Copies the coordinates into new buffers, reversing each of `ranges`.
/// `ranges` is iterated once per axis.
pub(crate) fn with_ranges_reversed(
    coords: &ValidCoords,
    ranges: impl Iterator<Item = Range<usize>> + Clone,
) -> ValidCoords {
    let reversed = |axis: &[f64]| {
        let mut axis = axis.to_vec();
        for range in ranges.clone() {
            axis[range].reverse();
        }
        axis
    };
    ValidCoords::from_axes((
        reversed(coords.x()),
        reversed(coords.y()),
        coords.z().map(reversed),
        coords.m().map(reversed),
    ))
}

/// Copies the coordinates at `ranges` into new buffers, in order.
/// `ranges` is iterated twice, once to size the buffers and once to copy.
pub(crate) fn copy_ranges(
    coords: &ValidCoords,
    ranges: impl Iterator<Item = Range<usize>> + Clone,
) -> ValidCoords {
    match (coords.z().is_some(), coords.m().is_some()) {
        (false, false) => copy_ranges_with::<false, false>(coords, ranges),
        (true, false) => copy_ranges_with::<true, false>(coords, ranges),
        (false, true) => copy_ranges_with::<false, true>(coords, ranges),
        (true, true) => copy_ranges_with::<true, true>(coords, ranges),
    }
}

/// [copy_ranges], copying z only if `Z=true`, and m only if `M=true`.
/// `coords` must have every axis that is copied.
#[inline(always)]
fn copy_ranges_with<const Z: bool, const M: bool>(
    coords: &ValidCoords,
    ranges: impl Iterator<Item = Range<usize>> + Clone,
) -> ValidCoords {
    let len = ranges.clone().map(|range| range.len()).sum();
    let (src_x, src_y) = (coords.x(), coords.y());
    let (src_z, src_m) = (
        coords.z().filter(|_| Z).unwrap_or_default(),
        coords.m().filter(|_| M).unwrap_or_default(),
    );
    let (mut x, mut y) = (Vec::with_capacity(len), Vec::with_capacity(len));
    let mut z = Vec::with_capacity(if Z { len } else { 0 });
    let mut m = Vec::with_capacity(if M { len } else { 0 });
    for range in ranges {
        x.extend_from_slice(&src_x[range.clone()]);
        y.extend_from_slice(&src_y[range.clone()]);
        if Z {
            z.extend_from_slice(&src_z[range.clone()]);
        }
        if M {
            m.extend_from_slice(&src_m[range]);
        }
    }
    ValidCoords::from_axes((x, y, Z.then_some(z), M.then_some(m)))
}

/// Copies the coordinates at `indices` into new buffers, in that order.
pub(crate) fn gathered(coords: &ValidCoords, indices: &[usize]) -> ValidCoords {
    let gather = |axis: &[f64]| indices.iter().map(|&i| axis[i]).collect::<Vec<_>>();
    ValidCoords::from_axes((
        gather(coords.x()),
        gather(coords.y()),
        coords.z().map(gather),
        coords.m().map(gather),
    ))
}

/// Marks the rows at `failed` as null, on top of those that already are.
pub(crate) fn with_null_at(rows: List, failed: &[usize]) -> List {
    let succeeded = succeeded(rows.len(), failed);
    let validity = combine_validities_and(rows.validity(), succeeded.as_ref());
    rows.with_validity(validity)
}
