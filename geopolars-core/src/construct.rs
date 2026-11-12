use geopolars_compute::inner_nulls::{
    propagate_inner_nulls_of_line_strings, propagate_inner_nulls_of_multi_line_strings,
    propagate_inner_nulls_of_multi_points, propagate_inner_nulls_of_multi_polygons,
    propagate_inner_nulls_of_polygons,
};
use geopolars_compute::native::{multi_polygons_have_only_rings, polygons_with_added_holes};
use geopolars_compute::with_const_flags;
use geopolars_views::{Kind, MultiPolygonArray, UncheckedStorage};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::parts::{
    closed_rings, lines_per_row, one_ring_per_row, rows_validity, validated_polygons, CloseRings,
    NullParts,
};
use crate::refused;

/// Moves every null value below a row of one chunk of a `kind` column's storage out to that row.
/// Implements the validation the Python constructors do once they have gathered the parts.
/// A (multi)polygon is also null where it does not have only rings,
/// after its open rings are closed with a `close`.
/// A point is refused, as the constructors always gather into lists.
pub fn validated_storage(
    chunk: &dyn Array,
    kind: Kind,
    close: Option<CloseRings>,
) -> PolarsResult<Box<dyn Array>> {
    let storage = UncheckedStorage::new(chunk)?;
    Ok(match kind {
        Kind::Point => return Err(refused("construct", kind)),
        Kind::LineString => propagate_inner_nulls_of_line_strings(storage)
            .into_list_array()
            .boxed(),
        Kind::MultiPoint => propagate_inner_nulls_of_multi_points(storage)
            .into_list_array()
            .boxed(),
        Kind::MultiLineString => propagate_inner_nulls_of_multi_line_strings(storage)
            .into_list_array()
            .boxed(),
        Kind::Polygon => validated_polygons(propagate_inner_nulls_of_polygons(storage), close),
        Kind::MultiPolygon => {
            let mut array = propagate_inner_nulls_of_multi_polygons(storage);
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
            .into_list_array()
            .boxed()
        }
    })
}

/// Assembles one chunk of polygons from one chunk of exterior rings and one of holes.
/// Implements the Python `polygons_from_rings_in_separate_columns`, and `polygons_from_coordinates`
/// without `holes`.
/// The holes only hold a single ring per row if `listed=false`.
pub fn assembled_polygons(
    exteriors: &dyn Array,
    holes: Option<(&dyn Array, bool, NullParts)>,
    close: Option<CloseRings>,
) -> PolarsResult<Box<dyn Array>> {
    let exteriors = one_ring_per_row(propagate_inner_nulls_of_line_strings(
        UncheckedStorage::new(exteriors)?,
    ));
    let Some((holes, listed, null_parts)) = holes else {
        // FIXME: this copies every coordinate into new buffers for nothing.
        // `exteriors` already is a trimmed polygon of one ring per row, without a copy,
        // so this can be `return Ok(validated_polygons(exteriors, close))`.
        // `polygons_with_added_holes` can then take `holes` without the `Option`.
        // This affects `polygons_from_coordinates`,
        // and `polygons_from_rings_in_separate_columns` without holes.
        let validity = exteriors.rows.validity().cloned();
        let polygons = polygons_with_added_holes(&exteriors, None, validity);
        return Ok(validated_polygons(polygons, close));
    };

    let holes = lines_per_row(holes, listed, null_parts)?;
    let validity = rows_validity(&exteriors.rows, holes.rows(), null_parts);
    let polygons = polygons_with_added_holes(&exteriors, Some(&holes), validity);
    Ok(validated_polygons(polygons, close))
}
