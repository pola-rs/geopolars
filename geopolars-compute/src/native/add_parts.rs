//! Adding the parts of one multi-geometry column to those of another.
//! Implements the Python `add_parts`.

use geopolars_views::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;

use super::concatenated::{concatenate_parts_onto_geometry, Built};
use super::coordinate_selection::{which_are_not_selected, CoordinateSelection, NonFinite};
use super::is_empty::{line_strings_are_empty, polygons_are_empty};
use crate::inner_nulls::MaskedParts;

fn keep_without_empty(keep: Option<Bitmap>, is_empty: BooleanArray) -> Option<Bitmap> {
    combine_validities_and(keep.as_ref(), Some(&!is_empty.values()))
}

/// Unsets the empty parts in the mask of a [MaskedParts].
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

/// [multi_line_strings_with_added_parts], for multipoints.
pub fn multi_points_with_added_parts(
    base_geometry: &MaskedParts<MultiPointArray>,
    parts_added: &MaskedParts<MultiPointArray>,
    output_validity: Option<Bitmap>,
) -> MultiPointArray {
    let Built { rows, coords, .. } = concatenate_parts_onto_geometry(
        (&base_geometry.view).into(),
        base_geometry.keep.as_ref(),
        Some((&parts_added.view).into()),
        parts_added.keep.as_ref(),
        output_validity,
    );
    MultiPointArray::from_parts(rows, coords)
}

/// Appends the linestrings of `parts_added` to the multilinestring on the same row of `base_geometry`.
/// Only the linestrings set in the masks of both are kept.
///
/// `base_geometry` and `parts_added` must have the same length and dimension.
/// `output_validity` must mark every null row of `base_geometry` as null.
pub fn multi_line_strings_with_added_parts(
    base_geometry: &MaskedParts<MultiLineStringArray>,
    parts_added: &MaskedParts<MultiLineStringArray>,
    output_validity: Option<Bitmap>,
) -> MultiLineStringArray {
    let Built {
        rows,
        levels: [lines],
        coords,
    } = concatenate_parts_onto_geometry(
        (&base_geometry.view).into(),
        base_geometry.keep.as_ref(),
        Some((&parts_added.view).into()),
        parts_added.keep.as_ref(),
        output_validity,
    );
    MultiLineStringArray::from_parts(rows, lines, coords)
}

/// [multi_line_strings_with_added_parts], for multipolygons.
pub fn multi_polygons_with_added_parts(
    base_geometry: &MaskedParts<MultiPolygonArray>,
    parts_added: &MaskedParts<MultiPolygonArray>,
    output_validity: Option<Bitmap>,
) -> MultiPolygonArray {
    let Built {
        rows,
        levels: [polygons, rings],
        coords,
    } = concatenate_parts_onto_geometry(
        (&base_geometry.view).into(),
        base_geometry.keep.as_ref(),
        Some((&parts_added.view).into()),
        parts_added.keep.as_ref(),
        output_validity,
    );
    MultiPolygonArray::from_parts(rows, polygons, rings, coords)
}
