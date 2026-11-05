//! Adding the parts of one multi-geometry column to those of another.
//! The parts are copied as they are; only `without_empty_parts` looks at them, to find empty ones.

use geopolars_core::{
    Coords, LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;

use super::concatenated::{concatenated, Built};
use super::is_empty::{line_strings_are_empty, points_are_empty, polygons_are_empty};
use crate::inner_nulls::MaskedParts;

/// `keep`, with every part that `empty` holds as `true` unset as well.
fn keep_without_empty(keep: Option<Bitmap>, empty: BooleanArray) -> Option<Bitmap> {
    combine_validities_and(keep.as_ref(), Some(&!empty.values()))
}

impl MaskedParts<MultiPointArray> {
    /// Unsets the empty points (every axis NaN) in the mask too.
    pub fn without_empty_parts(self) -> Self {
        let points = PointArray::new(Coords::from_valid(self.view.coords.clone(), None));
        let keep = keep_without_empty(self.keep, points_are_empty(&points));
        Self { keep, ..self }
    }
}

impl MaskedParts<MultiLineStringArray> {
    /// Unsets the linestrings without vertices in the mask too.
    pub fn without_empty_parts(self) -> Self {
        let lines = LineStringArray::from_parts(
            List::from_offsets(self.view.lines.offsets().clone(), None),
            self.view.coords.clone(),
        );
        let empty = line_strings_are_empty::<false>(&lines.into());
        let keep = keep_without_empty(self.keep, empty);
        Self { keep, ..self }
    }
}

impl MaskedParts<MultiPolygonArray> {
    /// Unsets the polygons without vertices in the mask too.
    pub fn without_empty_parts(self) -> Self {
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

/// `base`, each followed by the points on the same row of `added`.
/// See [multi_line_strings_with_added_parts].
pub fn multi_points_with_added_parts(
    base: &MultiPointArray,
    added: &MaskedParts<MultiPointArray>,
    validity: Option<Bitmap>,
) -> MultiPointArray {
    let keep = added.keep.as_ref();
    let Built { rows, coords, .. } =
        concatenated(base.into(), Some((&added.view).into()), keep, validity);
    MultiPointArray::from_parts(rows, coords)
}

/// `base`, each followed by the linestrings on the same row of `added`.
/// A null row of `added` adds nothing.
/// Only the linestrings of `added` set in its mask are added.
/// The rows are null where `validity` says.
/// Every null row of `base` has to be null in it too.
///
/// `base` and `added` have to have the same length and dimension.
pub fn multi_line_strings_with_added_parts(
    base: &MultiLineStringArray,
    added: &MaskedParts<MultiLineStringArray>,
    validity: Option<Bitmap>,
) -> MultiLineStringArray {
    let keep = added.keep.as_ref();
    let Built {
        rows,
        levels: [lines],
        coords,
    } = concatenated(base.into(), Some((&added.view).into()), keep, validity);
    MultiLineStringArray::from_parts(rows, lines, coords)
}

/// `base`, each followed by the polygons on the same row of `added`.
/// See [multi_line_strings_with_added_parts].
pub fn multi_polygons_with_added_parts(
    base: &MultiPolygonArray,
    added: &MaskedParts<MultiPolygonArray>,
    validity: Option<Bitmap>,
) -> MultiPolygonArray {
    let keep = added.keep.as_ref();
    let Built {
        rows,
        levels: [polygons, rings],
        coords,
    } = concatenated(base.into(), Some((&added.view).into()), keep, validity);
    MultiPolygonArray::from_parts(rows, polygons, rings, coords)
}
