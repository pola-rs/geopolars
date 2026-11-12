use geopolars_compute::native::{
    line_string_bounds, line_string_column_extent, multi_line_string_bounds,
    multi_line_string_column_extent, multi_line_string_part_bounds, multi_point_bounds,
    multi_point_column_extent, multi_point_part_bounds, multi_polygon_bounds,
    multi_polygon_column_extent, multi_polygon_part_bounds, point_bounds, point_column_extent,
    polygon_bounds, polygon_column_extent, BoxOptions, Extent,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, Coords, GeoDimension, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

/// Finds the bounding box of every geometry in one chunk of a `kind` column's storage
/// of dimension `dim`.
/// Implements the Python `bounds`.
/// A multi-geometry gets a list of boxes, one per part, if `per_part=true`.
pub fn bounds(
    chunk: &dyn Array,
    kind: Kind,
    dim: GeoDimension,
    per_part: bool,
    options: &BoxOptions,
) -> PolarsResult<Box<dyn Array>> {
    let (has_z, has_m) = (dim.has_z(), dim.has_m());
    Ok(match kind {
        Kind::Point => {
            let array = PointArray::new(Coords::new(chunk)?);
            with_const_flags!(Z = has_z, M = has_m; point_bounds::<Z, M>(&array, options)).boxed()
        }
        Kind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(
                Z = has_z, M = has_m;
                line_string_bounds::<Z, M>(&array, options)
            )
            .boxed()
        }
        Kind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(Z = has_z, M = has_m; polygon_bounds::<Z, M>(&array, options)).boxed()
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            if per_part {
                with_const_flags!(
                    Z = has_z, M = has_m;
                    multi_point_part_bounds::<Z, M>(&array, options)
                )
                .boxed()
            } else {
                with_const_flags!(
                    Z = has_z, M = has_m;
                    multi_point_bounds::<Z, M>(&array, options)
                )
                .boxed()
            }
        }
        Kind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            if per_part {
                with_const_flags!(
                    Z = has_z, M = has_m;
                    multi_line_string_part_bounds::<Z, M>(&array, options)
                )
                .boxed()
            } else {
                with_const_flags!(
                    Z = has_z, M = has_m;
                    multi_line_string_bounds::<Z, M>(&array, options)
                )
                .boxed()
            }
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            if per_part {
                with_const_flags!(
                    Z = has_z, M = has_m;
                    multi_polygon_part_bounds::<Z, M>(&array, options)
                )
                .boxed()
            } else {
                with_const_flags!(
                    Z = has_z, M = has_m;
                    multi_polygon_bounds::<Z, M>(&array, options)
                )
                .boxed()
            }
        }
    })
}

/// Finds the extent of every geometry in one chunk of a `kind` column's storage together.
/// Implements the Python `total_bounds` for one chunk.
pub fn column_extent<const Z: bool, const M: bool, const IGNORE_NULLS: bool>(
    chunk: &dyn Array,
    kind: Kind,
) -> PolarsResult<Option<Extent<Z, M>>> {
    Ok(match kind {
        Kind::Point => {
            point_column_extent::<Z, M, IGNORE_NULLS>(&PointArray::new(Coords::new(chunk)?))
        }
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            line_string_column_extent::<Z, M, IGNORE_NULLS>(&array)
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            polygon_column_extent::<Z, M, IGNORE_NULLS>(&array)
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            multi_point_column_extent::<Z, M, IGNORE_NULLS>(&array)
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            multi_line_string_column_extent::<Z, M, IGNORE_NULLS>(&array)
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            multi_polygon_column_extent::<Z, M, IGNORE_NULLS>(&array)
        }
    })
}
