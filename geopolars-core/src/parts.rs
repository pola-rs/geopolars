use geopolars_compute::inner_nulls::{
    multi_line_strings_with_null_parts_marked, multi_points_with_null_parts_marked,
    multi_polygons_with_null_parts_marked, propagate_inner_nulls_of_line_strings,
    propagate_inner_nulls_of_multi_line_strings, propagate_inner_nulls_of_multi_points,
    propagate_inner_nulls_of_multi_polygons, propagate_inner_nulls_of_points,
    propagate_inner_nulls_of_polygons, MaskedParts,
};
use geopolars_compute::native::{
    close_open_rings, multi_polygons_have_only_rings, polygons_have_only_rings,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, UncheckedStorage, ValidCoords, ValidList,
};
use polars_arrow::array::Array;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::Offsets;
use polars_core::prelude::PolarsResult;

/// How to close an open ring. See [`close_open_rings`] for `snap_z` and `snap_m`.
#[derive(Debug, Clone, Copy)]
pub struct CloseRings {
    pub snap_z: bool,
    pub snap_m: bool,
}

/// What a null part does to the geometry it is added to.
#[derive(Debug, Clone, Copy, Default)]
pub enum NullParts {
    /// The null part is left out.
    Drop,
    /// The null part makes the result null.
    #[default]
    Null,
}

/// See [`close_open_rings`].
/// The rings and their coordinates, or `None` if no ring changed.
pub(crate) fn closed_rings(
    rings: &ValidList,
    coords: &ValidCoords,
    closing: CloseRings,
) -> Option<(ValidList, ValidCoords)> {
    let CloseRings { snap_z, snap_m } = closing;
    let z = coords.dim().has_z();
    let closed = with_const_flags!(Z = z; close_open_rings::<Z>(rings, coords, snap_z, snap_m))?;
    Some((closed.rows.ignoring_validity().clone(), closed.coords))
}

/// `array` with its open rings closed first if there is a `close`,
/// and null where it does not have only rings (see [polygons_have_only_rings]).
pub(crate) fn validated_polygons(
    mut array: PolygonArray,
    close: Option<CloseRings>,
) -> PolygonArray {
    if let Some((rings, coords)) =
        close.and_then(|closing| closed_rings(&array.rings, &array.coords, closing))
    {
        array = PolygonArray::from_parts(array.rows, rings, coords);
    }
    // A null row is `false` in the values too.
    let z = array.coords.dim().has_z();
    let valid = with_const_flags!(Z = z; polygons_have_only_rings::<Z>(&array))
        .values()
        .clone();
    PolygonArray::from_parts(
        array.rows.with_validity(Some(valid)),
        array.rings,
        array.coords,
    )
}

/// See [validated_polygons].
pub(crate) fn validated_multi_polygons(
    mut array: MultiPolygonArray,
    close: Option<CloseRings>,
) -> MultiPolygonArray {
    if let Some((rings, coords)) =
        close.and_then(|closing| closed_rings(&array.rings, &array.coords, closing))
    {
        array = MultiPolygonArray::from_parts(array.rows, array.polygons, rings, coords);
    }
    // A null row is `false` in the values too.
    let z = array.coords.dim().has_z();
    let valid = with_const_flags!(Z = z; multi_polygons_have_only_rings::<Z>(&array))
        .values()
        .clone();
    MultiPolygonArray::from_parts(
        array.rows.with_validity(Some(valid)),
        array.polygons,
        array.rings,
        array.coords,
    )
}

fn one_per_row(len: usize, validity: Option<&Bitmap>) -> List {
    let lengths = std::iter::repeat_n(1, len);
    let offsets = Offsets::try_from_lengths(lengths).expect("one per row");
    List::from_offsets(offsets.into(), validity.cloned())
}

pub(crate) fn one_ring_per_row(exteriors: LineStringArray) -> PolygonArray {
    PolygonArray::from_parts(
        one_per_row(exteriors.rows.len(), exteriors.rows.validity()),
        exteriors.rows.ignoring_validity().clone(),
        exteriors.coords,
    )
}

pub(crate) fn one_ring_per_polygon(exteriors: MultiLineStringArray) -> MultiPolygonArray {
    MultiPolygonArray::from_parts(
        exteriors.rows,
        one_per_row(exteriors.lines.len(), None)
            .ignoring_validity()
            .clone(),
        exteriors.lines,
        exteriors.coords,
    )
}

pub(crate) fn one_point_per_row(points: PointArray) -> MultiPointArray {
    MultiPointArray::from_parts(
        one_per_row(points.coords.len(), points.coords.validity()),
        points.coords.ignoring_validity().clone(),
    )
}

pub(crate) fn one_line_per_row(lines: LineStringArray) -> MultiLineStringArray {
    MultiLineStringArray::from_parts(
        one_per_row(lines.rows.len(), lines.rows.validity()),
        lines.rows.ignoring_validity().clone(),
        lines.coords,
    )
}

pub(crate) fn one_polygon_per_row(polygons: PolygonArray) -> MultiPolygonArray {
    MultiPolygonArray::from_parts(
        one_per_row(polygons.rows.len(), polygons.rows.validity()),
        polygons.rows.ignoring_validity().clone(),
        polygons.rings,
        polygons.coords,
    )
}

/// Combines the validity of `base` and `parts` into that of a result built out of them.
/// A null row of `parts` only makes the result null with [NullParts::Null].
pub(crate) fn rows_validity(base: &List, parts: &List, null_parts: NullParts) -> Option<Bitmap> {
    match null_parts {
        NullParts::Drop => base.validity().cloned(),
        NullParts::Null => combine_validities_and(base.validity(), parts.validity()),
    }
}

/// Reads a chunk of linestrings as multilinestrings.
/// `chunk` only holds a single linestring per row if `listed=false`.
/// A linestring holding a null value makes its row null.
/// In a list of linestrings, [NullParts::Drop] unsets it in the mask instead.
pub(crate) fn lines_per_row(
    chunk: &dyn Array,
    listed: bool,
    null_parts: NullParts,
) -> PolarsResult<MaskedParts<MultiLineStringArray>> {
    let storage = UncheckedStorage::new(chunk)?;
    Ok(match (listed, null_parts) {
        (false, _) => one_line_per_row(propagate_inner_nulls_of_line_strings(storage)).into(),
        (true, NullParts::Drop) => multi_line_strings_with_null_parts_marked(storage),
        (true, NullParts::Null) => propagate_inner_nulls_of_multi_line_strings(storage).into(),
    })
}

/// [lines_per_row], for points.
pub(crate) fn points_per_row(
    chunk: &dyn Array,
    listed: bool,
    null_parts: NullParts,
) -> PolarsResult<MaskedParts<MultiPointArray>> {
    let storage = UncheckedStorage::new(chunk)?;
    Ok(match (listed, null_parts) {
        (false, _) => one_point_per_row(propagate_inner_nulls_of_points(storage)).into(),
        (true, NullParts::Drop) => multi_points_with_null_parts_marked(storage),
        (true, NullParts::Null) => propagate_inner_nulls_of_multi_points(storage).into(),
    })
}

/// [lines_per_row], for polygons.
pub(crate) fn polygons_per_row(
    chunk: &dyn Array,
    listed: bool,
    null_parts: NullParts,
) -> PolarsResult<MaskedParts<MultiPolygonArray>> {
    let storage = UncheckedStorage::new(chunk)?;
    Ok(match (listed, null_parts) {
        (false, _) => one_polygon_per_row(propagate_inner_nulls_of_polygons(storage)).into(),
        (true, NullParts::Drop) => multi_polygons_with_null_parts_marked(storage),
        (true, NullParts::Null) => propagate_inner_nulls_of_multi_polygons(storage).into(),
    })
}
