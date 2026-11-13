use geopolars_compute::native::close_open_rings;
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, GeometryArray, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::parts::{closed_rings, CloseRings};
use crate::SegmentKind;

/// Closes the open rings (and linestrings) in one chunk of a `kind` column's storage.
/// Implements the Python `close_rings`.
/// The result is of the same kind and dimension.
pub fn close_rings(
    chunk: &dyn Array,
    kind: SegmentKind,
    closing: CloseRings,
) -> PolarsResult<GeometryArray> {
    let CloseRings { snap_z, snap_m } = closing;
    Ok(match kind {
        SegmentKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            let z = array.coords.dim().has_z();
            let rows = array.rows.ignoring_validity();
            let closed = with_const_flags!(
                Z = z;
                close_open_rings::<Z>(rows, &array.coords, snap_z, snap_m)
            );
            GeometryArray::LineString(match closed {
                Some(closed) => LineStringArray::from_parts(
                    closed.rows.with_validity(array.rows.validity().cloned()),
                    closed.coords,
                ),
                None => array,
            })
        }
        SegmentKind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            GeometryArray::Polygon(match closed_rings(&array.rings, &array.coords, closing) {
                Some((rings, coords)) => PolygonArray::from_parts(array.rows, rings, coords),
                None => array,
            })
        }
        SegmentKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            GeometryArray::MultiLineString(
                match closed_rings(&array.lines, &array.coords, closing) {
                    Some((lines, coords)) => {
                        MultiLineStringArray::from_parts(array.rows, lines, coords)
                    }
                    None => array,
                },
            )
        }
        SegmentKind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            GeometryArray::MultiPolygon(match closed_rings(&array.rings, &array.coords, closing) {
                Some((rings, coords)) => {
                    MultiPolygonArray::from_parts(array.rows, array.polygons, rings, coords)
                }
                None => array,
            })
        }
    })
}
