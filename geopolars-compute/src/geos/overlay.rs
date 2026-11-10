//! The overlays of two geometries (their intersection or difference), forwarded to GEOS.
//! See `intersection` and `difference` in Python for the rules.

use geopolars_core::output::SizeHint;
use geopolars_core::{
    GeoDimension, GeometryArray, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use geos::{CoordType, Geom, Geometry, GeometryTypes};

use super::pairs::GeosPairs;
use super::{GeosGeometries, GeosRow};
use crate::{row_or_null, GeoComputeError, GeoComputeResult};

/// The topological dimension of the parts an overlay keeps.
#[derive(Debug, Clone, Copy)]
enum TopologicalDimension {
    Points,
    Lines,
    Polygons,
}

impl TopologicalDimension {
    fn single_and_multi_types(self) -> (GeometryTypes, GeometryTypes) {
        match self {
            Self::Points => (GeometryTypes::Point, GeometryTypes::MultiPoint),
            Self::Lines => (GeometryTypes::LineString, GeometryTypes::MultiLineString),
            Self::Polygons => (GeometryTypes::Polygon, GeometryTypes::MultiPolygon),
        }
    }

    fn create_multi(self, parts: Vec<Geometry>) -> GeoComputeResult<Geometry> {
        Ok(match self {
            Self::Points => Geometry::create_multipoint(parts)?,
            Self::Lines => Geometry::create_multiline_string(parts)?,
            Self::Polygons => Geometry::create_multipolygon(parts)?,
        })
    }
}

/// Which overlay [`OverlayPairs`] computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overlay {
    /// What `a` and `b` have in common.
    Intersection,
    /// What is left of `a` once `b` is taken away.
    Difference,
}

impl Overlay {
    fn pairs(
        self,
        a: &GeometryArray,
        b: &GeometryArray,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosPairs> {
        let (ignore, early) = (ignore_per_row_errors, early_finite_check);
        match self {
            Self::Intersection => GeosPairs::new_skipping_partners_of_empty(a, b, ignore, early),
            Self::Difference => GeosPairs::new_skipping_partners_of_empty_a(a, b, ignore, early),
        }
    }
}

/// Two arrays of geometries converted to GEOS, with their rows paired up to overlay.
/// A single row on either side is paired with every row of the other.
/// Each `into_*` method narrows every overlay down to its parts of one dimension,
/// and copies them, without m, into our array of that kind.
pub struct OverlayPairs {
    pairs: GeosPairs,
    overlay: Overlay,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

impl OverlayPairs {
    /// Converts both sides to GEOS and pairs up their rows,
    /// for the Python `intersection` or `difference` (which document their behaviour).
    /// A row whose overlay is already decided by a null or empty partner is not converted.
    /// An empty `b` makes the difference a copy of `a`, so that row of `a` is still converted, and can fail.
    /// A row that fails, here or in an `into_*` method, is null if `ignore_per_row_errors=true`.
    ///
    /// `a` and `b` must have the same length, or one of them a single row.
    pub fn new(
        a: &GeometryArray,
        b: &GeometryArray,
        overlay: Overlay,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<Self> {
        Ok(Self {
            pairs: overlay.pairs(a, b, ignore_per_row_errors, early_finite_check)?,
            overlay,
            ignore_per_row_errors,
            early_finite_check,
        })
    }

    /// A row with several points fails with [`GeoComputeError::Split`].
    pub fn into_points(self, z: bool) -> GeoComputeResult<PointArray> {
        let dim = if z {
            GeoDimension::XYZ
        } else {
            GeoDimension::XY
        };
        self.overlaid(TopologicalDimension::Points, false, false)?
            .into_point_array(dim)
    }

    pub fn into_multi_points(self, z: bool) -> GeoComputeResult<MultiPointArray> {
        let overlaid = self.overlaid(TopologicalDimension::Points, true, false)?;
        with_const_flags!(Z = z; overlaid.into_multi_point_array::<Z>(self.ignore_per_row_errors))
    }

    /// A row with several linestrings fails with [`GeoComputeError::Split`].
    pub fn into_line_strings(self, z: bool) -> GeoComputeResult<LineStringArray> {
        let overlaid = self.overlaid(TopologicalDimension::Lines, false, false)?;
        with_const_flags!(Z = z; overlaid.into_line_string_array::<Z>(self.ignore_per_row_errors))
    }

    pub fn into_multi_line_strings(self, z: bool) -> GeoComputeResult<MultiLineStringArray> {
        let overlaid = self.overlaid(TopologicalDimension::Lines, true, false)?;
        with_const_flags!(Z = z; overlaid.into_multi_line_string_array::<Z>(self.ignore_per_row_errors))
    }

    /// A row with several polygons fails with [`GeoComputeError::Split`].
    /// The rings are oriented as `ccw_outer` in the Python `intersection` and `difference` describes.
    pub fn into_polygons(self, z: bool, ccw_outer: bool) -> GeoComputeResult<PolygonArray> {
        self.overlaid(TopologicalDimension::Polygons, false, ccw_outer)?
            .into_polygon_array(z, self.ignore_per_row_errors, ccw_outer)
    }

    /// `ccw_outer`: see [`OverlayPairs::into_polygons`]
    pub fn into_multi_polygons(
        self,
        z: bool,
        ccw_outer: bool,
    ) -> GeoComputeResult<MultiPolygonArray> {
        self.overlaid(TopologicalDimension::Polygons, true, ccw_outer)?
            .into_multi_polygon_array(z, self.ignore_per_row_errors, ccw_outer)
    }

    /// Overlays every pair, and narrows each overlay down with [`kept_parts`].
    /// A row that copies `a` is reversed with `reverse_copies_of_a=true`,
    /// so that a builder that reverses every ring gives back `a` as it was.
    fn overlaid(
        &self,
        kept_dimension: TopologicalDimension,
        multi: bool,
        reverse_copies_of_a: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        let reverse = reverse_copies_of_a;
        let geometries = self
            .pairs
            .iter()
            .map(|(a, b)| self.row(a, b, kept_dimension, multi, reverse))
            .collect::<GeoComputeResult<Vec<_>>>()?;
        Ok(GeosGeometries {
            geometries,
            size_hint: summed_size_hint(self.pairs.size_hints()),
        })
    }

    /// Overlays one pair of rows from [`Overlay::pairs`], which decides the rows it sees as `Empty`.
    #[inline(always)]
    fn row(
        &self,
        a: &GeosRow,
        b: &GeosRow,
        kept_dimension: TopologicalDimension,
        multi: bool,
        reverse_copies_of_a: bool,
    ) -> GeoComputeResult<GeosRow> {
        let (ignore, early) = (self.ignore_per_row_errors, self.early_finite_check);
        match self.overlay {
            Overlay::Intersection => intersect_rows(a, b, kept_dimension, multi, ignore),
            Overlay::Difference => {
                let reverse = reverse_copies_of_a;
                difference_rows(a, b, kept_dimension, multi, reverse, ignore, early)
            }
        }
    }
}

fn summed_size_hint((a, b): (SizeHint<2>, SizeHint<2>)) -> SizeHint<2> {
    // Overlay adds a vertex wherever the two cross,
    // so the result can have more vertices than both inputs together.
    SizeHint {
        coords: a.coords + b.coords,
        parts: [a.parts[0] + b.parts[0], a.parts[1] + b.parts[1]],
    }
}

#[inline(always)]
fn intersect_rows(
    a: &GeosRow,
    b: &GeosRow,
    kept_dimension: TopologicalDimension,
    multi: bool,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<GeosRow> {
    let intersection = match (a, b) {
        (GeosRow::Null, _) | (_, GeosRow::Null) => return Ok(GeosRow::Null),
        // A row is also `Empty` when its partner is
        // (see `GeosPairs::new_skipping_partners_of_empty`).
        // Either way, their intersection is empty.
        (GeosRow::Empty, _) | (_, GeosRow::Empty) => return Ok(GeosRow::Empty),
        (GeosRow::Geometry(a), GeosRow::Geometry(b)) => a
            .intersection(b)
            .map_err(GeoComputeError::from)
            .and_then(|intersection| kept_parts(intersection, kept_dimension, multi)),
    };
    Ok(row_or_null(intersection.map(Some), ignore_per_row_errors)?.unwrap_or(GeosRow::Null))
}

#[inline(always)]
fn difference_rows(
    a: &GeosRow,
    b: &GeosRow,
    kept_dimension: TopologicalDimension,
    multi: bool,
    reverse_copies_of_a: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<GeosRow> {
    let difference = match (a, b) {
        (GeosRow::Null, _) | (_, GeosRow::Null) => return Ok(GeosRow::Null),
        (GeosRow::Empty, _) => return Ok(GeosRow::Empty),
        // The arms above handle a null or empty `a`,
        // which are the only other reasons to take `b` as empty
        // (see `GeosPairs::new_skipping_partners_of_empty_a`).
        // This `b` is therefore empty itself, and takes nothing away.
        // The difference is `a` as it is, so it is copied without computing a difference.
        // The copy fits the output kind, as the `into_multi_*` methods take a single geometry as one part.
        // A copy is needed, as the pairs only lend their rows, and a single `a` is shared by every row.
        (GeosRow::Geometry(a), GeosRow::Empty) => {
            // A copy refuses a non-finite x or y whatever `early_finite_check` is,
            // so that the option never changes a copied row.
            // With `early_finite_check=true`, converting `a` has already done so.
            // GEOS itself does not fail on every such vertex
            // (a line minus a polygon it does not touch keeps a NaN),
            // so a copied row can be stricter than one GEOS computes.
            let finite = if early_finite_check {
                Ok(())
            } else {
                difference_rows_xy_is_finite_or_error(a)
            };
            let copy = || match reverse_copies_of_a {
                true => a.reverse(),
                false => Geom::clone(a),
            };
            finite.and_then(|()| Ok(GeosRow::Geometry(copy()?)))
        }
        // TODO: Also give `a` as it is when `b` does not touch it, as for an empty `b`.
        // GEOS now re-orients and nodes `a` even then, so it can differ from a copy, and even split.
        (GeosRow::Geometry(a), GeosRow::Geometry(b)) => a
            .difference(b)
            .map_err(GeoComputeError::from)
            .and_then(|difference| kept_parts(difference, kept_dimension, multi)),
    };
    Ok(row_or_null(difference.map(Some), ignore_per_row_errors)?.unwrap_or(GeosRow::Null))
}

fn difference_rows_xy_is_finite_or_error(geometry: &impl Geom) -> GeoComputeResult<()> {
    if geometry.is_empty()? {
        return Ok(());
    }
    match geometry.geometry_type()? {
        GeometryTypes::Point | GeometryTypes::LineString | GeometryTypes::LinearRing => {
            let xy = geometry.get_coord_seq()?.as_buffer(Some(CoordType::XY))?;
            let not_finite = xy
                .as_chunks::<2>()
                .0
                .iter()
                .find(|[x, y]| !(x.is_finite() && y.is_finite()));
            match not_finite {
                Some(&[x, y]) => Err(GeoComputeError::GeosNotFinite { position: (x, y) }),
                None => Ok(()),
            }
        }
        GeometryTypes::Polygon => {
            difference_rows_xy_is_finite_or_error(&geometry.get_exterior_ring()?)?;
            for n in 0..geometry.get_num_interior_rings()? {
                difference_rows_xy_is_finite_or_error(&geometry.get_interior_ring_n(n)?)?;
            }
            Ok(())
        }
        _ => {
            for n in 0..geometry.get_num_geometries()? {
                difference_rows_xy_is_finite_or_error(&geometry.get_geometry_n(n)?)?;
            }
            Ok(())
        }
    }
}

/// Narrows an overlay down to its parts of `kept_dimension`,
/// as one geometry for the `into_*_array` of that kind.
/// Several parts fail with [`GeoComputeError::Split`], unless `multi=true` puts them in a multi geometry.
fn kept_parts(
    overlay: Geometry,
    kept_dimension: TopologicalDimension,
    multi: bool,
) -> GeoComputeResult<GeosRow> {
    if overlay.is_empty()? {
        return Ok(GeosRow::Empty);
    }
    let (single_type, multi_type) = kept_dimension.single_and_multi_types();
    let found = overlay.geometry_type()?;
    // The common cases: GEOS gave the kept dimension only, so nothing is copied.
    if found == single_type || (found == multi_type && multi) {
        return Ok(GeosRow::Geometry(overlay));
    }
    if found != multi_type && found != GeometryTypes::GeometryCollection {
        return Ok(GeosRow::Empty);
    }
    // What is left is a collection, which can mix dimensions,
    // or a multi geometry that has to become a single one, or be refused if it has several parts.
    // Only these copy their kept parts.
    let mut kept = Vec::new();
    for n in 0..overlay.get_num_geometries()? {
        let part = overlay.get_geometry_n(n)?;
        if part.geometry_type()? == single_type && !part.is_empty()? {
            kept.push(Geom::clone(&part)?);
        }
    }
    match kept.len() {
        0 => Ok(GeosRow::Empty),
        _ if multi => Ok(GeosRow::Geometry(kept_dimension.create_multi(kept)?)),
        1 => Ok(GeosRow::Geometry(kept.pop().expect("one part"))),
        // This counts the parts as GEOS gives them.
        // WARNING: Only since GEOS 3.15 are line pieces that join end to end
        // merged into one linestring.
        // Before, a line that runs through a node (such as a vertex of the other boundary)
        // came back in pieces, so `LINESTRING (0 0, 2 0, 4 0)` with itself would split in two.
        // Even now, pieces that meet where three or more lines do
        // (such as where a line crosses itself) stay apart.
        // We get 3.15 from the `static` feature, which builds the GEOS bundled with `geos-src`
        // (its version is locked in Cargo.lock).
        // The `v3_15_0` feature only selects the bindings, and does not check the version.
        count => Err(GeoComputeError::Split { parts: count }),
    }
}
