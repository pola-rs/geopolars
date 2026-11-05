//! Moving every null value below a row out to the row,
//! so that the views hold to GeoArrow's rule on null values:
//! nulls only at the outermost level.
//! Of a multi-geometry, the parts with a null value can be marked in a mask instead.
//! They stay in the view with arbitrary values,
//! which is wrapped in a [MaskedParts] so that only a kernel taking the mask reads it.
//! This is a separate pass, before any kernel reads the views.

use geopolars_core::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, UncheckedStorage, ValidCoords, ValidList,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;

use crate::passing::rows_passing;

/// The validity of the rows of `storage`, with every null value below a row moved out to it:
/// a row is also null where a part or coordinate (or one of its axes) below it is.
/// `None` when every row is there.
fn propagate_inner_nulls(storage: &UncheckedStorage) -> Option<Bitmap> {
    let lists: Vec<&ValidList> = storage.lists.iter().map(List::ignoring_validity).collect();
    let validities = (storage.lists.iter().map(List::validity)).chain([storage.coords.validity()]);
    rows_passing(
        &lists,
        validities.map(|validity| validity.cloned()).collect(),
    )
}

/// The `N` list levels of `storage`, the rows with the validity of [propagate_inner_nulls],
/// and its coordinates.
/// Storage with another number of list levels is a bug.
fn propagate_inner_nulls_of_lists<const N: usize>(
    storage: UncheckedStorage,
) -> ([List; N], ValidCoords) {
    let rows = propagate_inner_nulls(&storage);
    let (mut lists, coords) = levels_of(storage);
    lists[0] = lists[0].clone().with_validity(rows);
    (lists, coords)
}

/// The `N` list levels of `storage` with their validity, and its coordinates without theirs.
/// Storage with another number of list levels is a bug.
fn levels_of<const N: usize>(storage: UncheckedStorage) -> ([List; N], ValidCoords) {
    let coords = storage.coords.ignoring_validity().clone();
    let levels = storage.lists.len();
    let lists: [List; N] = storage
        .lists
        .try_into()
        .unwrap_or_else(|_| panic!("expected {N} list levels, got {levels}"));
    (lists, coords)
}

/// Which parts of the rows of `storage` have no null value in or below them.
/// `None` when every part does.
fn parts_passing(storage: &UncheckedStorage) -> Option<Bitmap> {
    let below_parts = &storage.lists[1..];
    let lists: Vec<&ValidList> = below_parts.iter().map(List::ignoring_validity).collect();
    let validities = (below_parts.iter().map(List::validity)).chain([storage.coords.validity()]);
    rows_passing(
        &lists,
        validities.map(|validity| validity.cloned()).collect(),
    )
}

/// The points of `storage`, null where one of their axes is.
/// The other `propagate_inner_nulls_of_*` functions do the same for their kind:
/// a row is null where anything below it is.
pub fn propagate_inner_nulls_of_points(storage: UncheckedStorage) -> PointArray {
    let levels = storage.lists.len();
    assert_eq!(levels, 0, "expected points, got {levels} list levels");
    // The coordinates already combine the validity of the struct and its axes.
    PointArray::new(storage.coords)
}

pub fn propagate_inner_nulls_of_multi_points(storage: UncheckedStorage) -> MultiPointArray {
    let ([rows], coords) = propagate_inner_nulls_of_lists(storage);
    MultiPointArray::from_parts(rows, coords)
}

pub fn propagate_inner_nulls_of_line_strings(storage: UncheckedStorage) -> LineStringArray {
    let ([rows], coords) = propagate_inner_nulls_of_lists(storage);
    LineStringArray::from_parts(rows, coords)
}

pub fn propagate_inner_nulls_of_multi_line_strings(
    storage: UncheckedStorage,
) -> MultiLineStringArray {
    let ([rows, lines], coords) = propagate_inner_nulls_of_lists(storage);
    MultiLineStringArray::from_parts(rows, lines.ignoring_validity().clone(), coords)
}

pub fn propagate_inner_nulls_of_polygons(storage: UncheckedStorage) -> PolygonArray {
    let ([rows, rings], coords) = propagate_inner_nulls_of_lists(storage);
    PolygonArray::from_parts(rows, rings.ignoring_validity().clone(), coords)
}

pub fn propagate_inner_nulls_of_multi_polygons(storage: UncheckedStorage) -> MultiPolygonArray {
    let ([rows, polygons, rings], coords) = propagate_inner_nulls_of_lists(storage);
    MultiPolygonArray::from_parts(
        rows,
        polygons.ignoring_validity().clone(),
        rings.ignoring_validity().clone(),
        coords,
    )
}

/// The multipoints of `storage`, with only the null rows null.
/// The points without a null value are set in the mask (see [parts_passing]).
/// The other points stay in the view, with arbitrary values.
/// The other `*_with_null_parts_marked` functions do the same for their kind.
pub fn multi_points_with_null_parts_marked(
    storage: UncheckedStorage,
) -> MaskedParts<MultiPointArray> {
    let keep = parts_passing(&storage);
    let ([rows], coords) = levels_of(storage);
    MaskedParts {
        view: MultiPointArray::from_parts(rows, coords),
        keep,
    }
}

pub fn multi_line_strings_with_null_parts_marked(
    storage: UncheckedStorage,
) -> MaskedParts<MultiLineStringArray> {
    let keep = parts_passing(&storage);
    let ([rows, lines], coords) = levels_of(storage);
    let view = MultiLineStringArray::from_parts(rows, lines.ignoring_validity().clone(), coords);
    MaskedParts { view, keep }
}

pub fn multi_polygons_with_null_parts_marked(
    storage: UncheckedStorage,
) -> MaskedParts<MultiPolygonArray> {
    let keep = parts_passing(&storage);
    let ([rows, polygons, rings], coords) = levels_of(storage);
    let view = MultiPolygonArray::from_parts(
        rows,
        polygons.ignoring_validity().clone(),
        rings.ignoring_validity().clone(),
        coords,
    );
    MaskedParts { view, keep }
}

/// A view and a mask of the parts of it to read.
/// The parts with a null value in the input are unset in it,
/// and after [without_empty_parts](crate::native::WithoutEmptyParts::without_empty_parts) the empty ones too.
/// The unset parts may hold arbitrary values,
/// so the view is not handed out as a plain one.
// Only the `native` kernels read it.
#[cfg_attr(not(feature = "native"), allow(dead_code))]
pub struct MaskedParts<T> {
    pub(crate) view: T,
    /// One bit per part. `None` when every part is set.
    pub(crate) keep: Option<Bitmap>,
}

/// A plain view has no null values below its rows, so every part is set.
impl<T> From<T> for MaskedParts<T> {
    fn from(view: T) -> Self {
        Self { view, keep: None }
    }
}

impl MaskedParts<MultiPointArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.view.rows
    }
}

impl MaskedParts<MultiLineStringArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.view.rows
    }

    /// Makes the rows unset in `rows` null too.
    pub fn only_on_rows(mut self, rows: &Bitmap) -> Self {
        let validity = combine_validities_and(self.view.rows.validity(), Some(rows));
        self.view.rows = self.view.rows.with_validity(validity);
        self
    }
}

impl MaskedParts<MultiPolygonArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.view.rows
    }
}
