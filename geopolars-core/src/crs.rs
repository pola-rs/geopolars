use geopolars_compute::proj::{reproject_coords, reproject_valid_coords, Transformation};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// Reprojects every geometry in one chunk of a `kind` column's storage by `transformation`.
/// Implements the Python `to_crs`.
/// The result is the storage of the same kind and dimension.
///
/// `z`: see [`geopolars_compute::proj`]
pub fn reproject(
    chunk: &dyn Array,
    kind: Kind,
    transformation: &Transformation,
    z: bool,
) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        Kind::Point => {
            let coords = reproject_coords(transformation, &Coords::new(chunk)?, z).into_polars()?;
            PointArray::new(coords).into_struct_array().boxed()
        }
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            let coords = reproject_valid_coords(transformation, &array.coords, z).into_polars()?;
            LineStringArray::from_parts(array.rows, coords)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            let coords = reproject_valid_coords(transformation, &array.coords, z).into_polars()?;
            MultiPointArray::from_parts(array.rows, coords)
                .into_list_array()
                .boxed()
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            let coords = reproject_valid_coords(transformation, &array.coords, z).into_polars()?;
            PolygonArray::from_parts(array.rows, array.rings, coords)
                .into_list_array()
                .boxed()
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            let coords = reproject_valid_coords(transformation, &array.coords, z).into_polars()?;
            MultiLineStringArray::from_parts(array.rows, array.lines, coords)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            let coords = reproject_valid_coords(transformation, &array.coords, z).into_polars()?;
            MultiPolygonArray::from_parts(array.rows, array.polygons, array.rings, coords)
                .into_list_array()
                .boxed()
        }
    })
}
