//! Adding the parts of one multi-geometry column to those of another.

use geopolars_core::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;

use super::concatenated::{concatenated, Built};
use super::coordinate_selection::{which_are_not_selected, CoordinateSelection, NonFinite};
use super::is_empty::{line_strings_are_empty, polygons_are_empty};
use crate::inner_nulls::MaskedParts;

/// `keep`, with every part that `empty` holds as `true` unset as well.
fn keep_without_empty(keep: Option<Bitmap>, empty: BooleanArray) -> Option<Bitmap> {
    combine_validities_and(keep.as_ref(), Some(&!empty.values()))
}

/// Enables unsetting the empty parts in the mask of a [MaskedParts]
/// (see the README on `MaskedParts`).
pub trait WithoutEmptyParts {
    fn without_empty_parts(self) -> Self;
}

const EMPTY_POINTS: CoordinateSelection = CoordinateSelection {
    non_finite: Some(NonFinite {
        select_non_finite: false,
        select_empty: true,
        ignore_z: false,
        ignore_m: false,
    }),
    bounds: None,
};

impl WithoutEmptyParts for MaskedParts<MultiPointArray> {
    /// The empty points have every axis NaN.
    fn without_empty_parts(self) -> Self {
        let not_empty = which_are_not_selected(&self.view.coords, &EMPTY_POINTS);
        let keep = combine_validities_and(self.keep.as_ref(), not_empty.as_ref());
        Self { keep, ..self }
    }
}

impl WithoutEmptyParts for MaskedParts<MultiLineStringArray> {
    /// The empty linestrings have no vertices.
    fn without_empty_parts(self) -> Self {
        let lines = LineStringArray::from_parts(
            List::from_offsets(self.view.lines.offsets().clone(), None),
            self.view.coords.clone(),
        );
        let empty = line_strings_are_empty::<false>(&lines.into());
        let keep = keep_without_empty(self.keep, empty);
        Self { keep, ..self }
    }
}

impl WithoutEmptyParts for MaskedParts<MultiPolygonArray> {
    /// The empty polygons have no vertices.
    fn without_empty_parts(self) -> Self {
        let polygons = PolygonArray::from_parts(
            List::from_offsets(self.view.polygons.offsets().clone(), None),
            self.view.rings.clone(),
            self.view.coords.clone(),
        );
        let empty = polygons_are_empty::<false>(&polygons.into());
        let keep = keep_without_empty(self.keep, empty);
        Self { keep, ..self }
    }
}

pub fn multi_points_with_added_parts(
    base: &MaskedParts<MultiPointArray>,
    added: &MaskedParts<MultiPointArray>,
    validity: Option<Bitmap>,
) -> MultiPointArray {
    let Built { rows, coords, .. } = concatenated(
        (&base.view).into(),
        base.keep.as_ref(),
        Some((&added.view).into()),
        added.keep.as_ref(),
        validity,
    );
    MultiPointArray::from_parts(rows, coords)
}

/// `base`, each followed by the linestrings on the same row of `added`.
/// A null row of `added` adds nothing.
/// Only the linestrings of `base` and `added` set in their masks are taken.
/// The rows are null where `validity` says.
/// Every null row of `base` has to be null in it too.
///
/// `base` and `added` have to have the same length and dimension.
pub fn multi_line_strings_with_added_parts(
    base: &MaskedParts<MultiLineStringArray>,
    added: &MaskedParts<MultiLineStringArray>,
    validity: Option<Bitmap>,
) -> MultiLineStringArray {
    let Built {
        rows,
        levels: [lines],
        coords,
    } = concatenated(
        (&base.view).into(),
        base.keep.as_ref(),
        Some((&added.view).into()),
        added.keep.as_ref(),
        validity,
    );
    MultiLineStringArray::from_parts(rows, lines, coords)
}

/// `base`, each followed by the polygons on the same row of `added`.
/// See [multi_line_strings_with_added_parts].
pub fn multi_polygons_with_added_parts(
    base: &MaskedParts<MultiPolygonArray>,
    added: &MaskedParts<MultiPolygonArray>,
    validity: Option<Bitmap>,
) -> MultiPolygonArray {
    let Built {
        rows,
        levels: [polygons, rings],
        coords,
    } = concatenated(
        (&base.view).into(),
        base.keep.as_ref(),
        Some((&added.view).into()),
        added.keep.as_ref(),
        validity,
    );
    MultiPolygonArray::from_parts(rows, polygons, rings, coords)
}
