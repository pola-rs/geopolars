use geopolars_compute::inner_nulls::MaskedParts;
use geopolars_compute::native::{
    multi_line_strings_with_added_parts, multi_points_with_added_parts,
    multi_polygons_with_added_parts, WithoutEmptyParts,
};
use geopolars_views::{
    downcast, Coords, GeometryArray, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::parts::{
    lines_per_row, one_line_per_row, one_point_per_row, one_polygon_per_row, points_per_row,
    polygons_per_row, rows_validity, NullParts,
};

/// `empty_parts` of the Python `add_parts`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmptyParts {
    Keep,
    Drop,
}

/// Adds `parts` to every geometry in `base_geometry`, one chunk of a `kind` column's storage.
/// Implements the Python `add_parts`.
/// `parts` only holds a single part per row if `listed=false`.
pub fn add_parts(
    base_geometry: &dyn Array,
    kind: Kind,
    parts: &dyn Array,
    listed: bool,
    null_parts: NullParts,
    empty_parts: EmptyParts,
) -> PolarsResult<GeometryArray> {
    let rows = |what| downcast(base_geometry, what);
    let drop = empty_parts == EmptyParts::Drop;
    Ok(match kind {
        Kind::Point | Kind::MultiPoint => {
            let base = match kind {
                Kind::Point => without_empty_parts_if(
                    drop,
                    MaskedParts::from(one_point_per_row(PointArray::new(Coords::new(
                        base_geometry,
                    )?))),
                ),
                _ => MultiPointArray::new(rows("a list of coordinates")?)?.into(),
            };
            let parts = points_per_row(parts, listed, null_parts)?;
            let parts = without_empty_parts_if(drop, parts);
            let validity = rows_validity(base.rows(), parts.rows(), null_parts);
            GeometryArray::MultiPoint(multi_points_with_added_parts(&base, &parts, validity))
        }
        Kind::LineString | Kind::MultiLineString => {
            let base = match kind {
                Kind::LineString => {
                    let lines = LineStringArray::new(rows("a list of coordinates")?)?;
                    without_empty_parts_if(drop, MaskedParts::from(one_line_per_row(lines)))
                }
                _ => MultiLineStringArray::new(rows("a list of linestrings")?)?.into(),
            };
            let parts = lines_per_row(parts, listed, null_parts)?;
            let parts = without_empty_parts_if(drop, parts);
            let validity = rows_validity(base.rows(), parts.rows(), null_parts);
            let added = multi_line_strings_with_added_parts(&base, &parts, validity);
            GeometryArray::MultiLineString(added)
        }
        Kind::Polygon | Kind::MultiPolygon => {
            let base = match kind {
                Kind::Polygon => {
                    let polygons = PolygonArray::new(rows("a list of rings")?)?;
                    without_empty_parts_if(drop, MaskedParts::from(one_polygon_per_row(polygons)))
                }
                _ => MultiPolygonArray::new(rows("a list of polygons")?)?.into(),
            };
            let parts = polygons_per_row(parts, listed, null_parts)?;
            let parts = without_empty_parts_if(drop, parts);
            let validity = rows_validity(base.rows(), parts.rows(), null_parts);
            let added = multi_polygons_with_added_parts(&base, &parts, validity);
            GeometryArray::MultiPolygon(added)
        }
    })
}

fn without_empty_parts_if<T: WithoutEmptyParts>(drop: bool, parts: T) -> T {
    if drop {
        parts.without_empty_parts()
    } else {
        parts
    }
}
