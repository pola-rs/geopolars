use geopolars_compute::inner_nulls::{
    propagate_inner_nulls_of_line_strings, propagate_inner_nulls_of_multi_line_strings,
    propagate_inner_nulls_of_multi_points, propagate_inner_nulls_of_multi_polygons,
    propagate_inner_nulls_of_polygons,
};
use geopolars_compute::native::polygons_with_added_holes;
use geopolars_views::{GeometryArray, UncheckedStorage};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::parts::{
    lines_per_row, one_ring_per_polygon, one_ring_per_row, rows_validity, validated_multi_polygons,
    validated_polygons, CloseRings, NullParts,
};
use crate::NestedKind;

/// Moves every null value below a row of one chunk of a `kind` column's storage out to that row.
/// Implements the validation the Python constructors do once they have gathered the parts.
/// A (multi)polygon is also null where it does not have only rings,
/// after its open rings are closed with a `close`.
pub fn validated_storage(
    chunk: &dyn Array,
    kind: NestedKind,
    close: Option<CloseRings>,
) -> PolarsResult<GeometryArray> {
    let storage = UncheckedStorage::new(chunk)?;
    Ok(match kind {
        NestedKind::LineString => {
            GeometryArray::LineString(propagate_inner_nulls_of_line_strings(storage))
        }
        NestedKind::MultiPoint => {
            GeometryArray::MultiPoint(propagate_inner_nulls_of_multi_points(storage))
        }
        NestedKind::MultiLineString => {
            GeometryArray::MultiLineString(propagate_inner_nulls_of_multi_line_strings(storage))
        }
        NestedKind::Polygon => GeometryArray::Polygon(validated_polygons(
            propagate_inner_nulls_of_polygons(storage),
            close,
        )),
        NestedKind::MultiPolygon => GeometryArray::MultiPolygon(validated_multi_polygons(
            propagate_inner_nulls_of_multi_polygons(storage),
            close,
        )),
    })
}

/// Assembles one chunk of polygons from one chunk of exterior rings and one of holes.
/// Implements the Python `polygons_from_rings_in_separate_columns`,
/// and `polygons_from_coordinates` without `holes`.
/// The holes only hold a single ring per row if `listed=false`.
pub fn assembled_polygons(
    exteriors: &dyn Array,
    holes: Option<(&dyn Array, bool, NullParts)>,
    close: Option<CloseRings>,
) -> PolarsResult<GeometryArray> {
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
        return Ok(GeometryArray::Polygon(validated_polygons(polygons, close)));
    };

    let holes = lines_per_row(holes, listed, null_parts)?;
    let validity = rows_validity(&exteriors.rows, holes.rows(), null_parts);
    let polygons = polygons_with_added_holes(&exteriors, Some(&holes), validity);
    Ok(GeometryArray::Polygon(validated_polygons(polygons, close)))
}

/// Assembles one chunk of multipolygons from one chunk of rows of exterior rings,
/// with a polygon per ring.
/// Implements the Python `multi_polygons_from_coordinates`.
pub fn assembled_multi_polygons(
    exteriors: &dyn Array,
    close: Option<CloseRings>,
) -> PolarsResult<GeometryArray> {
    let exteriors = one_ring_per_polygon(propagate_inner_nulls_of_multi_line_strings(
        UncheckedStorage::new(exteriors)?,
    ));
    Ok(GeometryArray::MultiPolygon(validated_multi_polygons(
        exteriors, close,
    )))
}
