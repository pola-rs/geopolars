use geopolars_compute::native::close_open_rings;
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, Kind, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::parts::{closed_rings, CloseRings};
use crate::refused;

/// Closes the open rings (and linestrings) in one chunk of a `kind` column's storage.
/// Implements the Python `close_rings`.
/// The result is the storage of the same kind and dimension,
/// or `None` if no ring changed.
pub fn close_rings(
    chunk: &dyn Array,
    kind: Kind,
    closing: CloseRings,
) -> PolarsResult<Option<Box<dyn Array>>> {
    let CloseRings { snap_z, snap_m } = closing;
    Ok(match kind {
        Kind::Point | Kind::MultiPoint => return Err(refused("close_rings", kind)),
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            let z = array.coords.dim().has_z();
            let rows = array.rows.ignoring_validity();
            with_const_flags!(
                Z = z;
                close_open_rings::<Z>(rows, &array.coords, snap_z, snap_m)
            )
            .map(|closed| {
                LineStringArray::from_parts(
                    closed.rows.with_validity(array.rows.validity().cloned()),
                    closed.coords,
                )
                .into_list_array()
                .boxed()
            })
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            closed_rings(&array.rings, &array.coords, closing).map(|(rings, coords)| {
                PolygonArray::from_parts(array.rows, rings, coords)
                    .into_list_array()
                    .boxed()
            })
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            closed_rings(&array.lines, &array.coords, closing).map(|(lines, coords)| {
                MultiLineStringArray::from_parts(array.rows, lines, coords)
                    .into_list_array()
                    .boxed()
            })
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            closed_rings(&array.rings, &array.coords, closing).map(|(rings, coords)| {
                MultiPolygonArray::from_parts(array.rows, array.polygons, rings, coords)
                    .into_list_array()
                    .boxed()
            })
        }
    })
}
