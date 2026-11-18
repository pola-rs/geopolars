//! The overlays of two geometries (their intersection or difference), forwarded to GEOS.
//! Implements the Python `intersection` and `difference`.

use geopolars_views::output::SizeHint;
use geopolars_views::{
    GeoDimension, GeometryArray, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use geos::{CoordType, Geom, Geometry, GeometryTypes};

use super::pairs::{GeosPairs, PairOperation, WhenEmpty};
use super::{GeometryOrientation, GeosGeometries, GeosRow, GeosRows};
use crate::{
    row_or_null, EarlyCoordCheck, GeoComputeError, GeoComputeResult, NewHoles, NewRingOrientation,
    OnRowError,
};

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

/// The overlay of a pair, before [`kept_parts`] narrows it down to the output kind.
/// An empty row gives an empty overlay, except an empty `b` in a difference,
/// which takes nothing away from `a`.
struct OverlayOperation {
    overlay: Overlay,
    early_coord_check: EarlyCoordCheck,
}

impl PairOperation for OverlayOperation {
    type Output = GeosRow;

    fn when_both_empty(&self) -> GeosRow {
        GeosRow::Empty
    }

    fn when_a_empty(&self) -> WhenEmpty<Self> {
        WhenEmpty::Gives(|| GeosRow::Empty)
    }

    fn when_b_empty(&self) -> WhenEmpty<Self> {
        match self.overlay {
            Overlay::Intersection => WhenEmpty::Gives(|| GeosRow::Empty),
            Overlay::Difference => WhenEmpty::WithPartner(Self::unchanged_a),
        }
    }

    // TODO: Also give `a` as it is when `b` does not touch it, as for an empty `b`.
    // GEOS now re-orients and nodes `a` even then, so it can differ from a copy, and even split.
    fn on_geometries(&self, a: &Geometry, b: &Geometry) -> GeoComputeResult<GeosRow> {
        let overlay = match self.overlay {
            Overlay::Intersection => a.intersection(b)?,
            Overlay::Difference => a.difference(b)?,
        };
        Ok(GeosRow::Geometry {
            geometry: overlay,
            maybe_wrong_orientation: false,
        })
    }
}

impl OverlayOperation {
    /// The difference of `a` and an empty `b`: `a` as it is, copied without computing a difference.
    /// A copy is needed, as the pairs only lend their rows, and a single `a` is shared by every row.
    fn unchanged_a(&self, a: &Geometry) -> GeoComputeResult<GeosRow> {
        // A copy refuses a non-finite x or y whatever `early_coord_check` is,
        // so that the option never changes a copied row.
        // With `early_coord_check=`[EarlyCoordCheck::FiniteAndEmpty],
        // converting `a` has already done so.
        // [EarlyCoordCheck::Finite] lets an empty point through to GEOS,
        // but a copy never goes there, so it is refused here.
        // GEOS itself does not fail on every such vertex
        // (a line minus a polygon it does not touch keeps a NaN),
        // so a copied row can be stricter than one GEOS computes.
        if self.early_coord_check != EarlyCoordCheck::FiniteAndEmpty {
            xy_is_finite_or_error(a)?;
        }
        Ok(GeosRow::Geometry {
            geometry: Geom::clone(a)?,
            maybe_wrong_orientation: true,
        })
    }
}

/// Two arrays of geometries converted to GEOS, with their rows paired up to overlay.
/// A single row on either side is paired with every row of the other.
/// Each `into_*` method narrows every overlay down to its parts of one dimension,
/// and copies them, without m, into our array of that kind.
pub struct OverlayPairs {
    pairs: GeosPairs,
    operation: OverlayOperation,
    on_row_error: OnRowError,
}

impl OverlayPairs {
    /// Converts both sides to GEOS and pairs up their rows,
    /// for the Python `intersection` or `difference` (which document their behaviour).
    /// A row whose overlay is already decided by a null or empty partner is not converted.
    /// An empty `b` makes the difference a copy of `a`, so that row of `a` is still converted, and can fail.
    /// A row that fails, here or in an `into_*` method, goes to `on_row_error`.
    ///
    /// `a` and `b` must have the same length, or one of them a single row.
    pub fn new(
        a: &GeometryArray,
        b: &GeometryArray,
        overlay: Overlay,
        on_row_error: OnRowError,
        early_coord_check: EarlyCoordCheck,
    ) -> GeoComputeResult<Self> {
        let operation = OverlayOperation {
            overlay,
            early_coord_check,
        };
        Ok(Self {
            pairs: GeosPairs::new(a, b, &operation, on_row_error, early_coord_check)?,
            operation,
            on_row_error,
        })
    }

    /// A row with several points fails with [`GeoComputeError::Split`].
    pub fn into_points(self, z: bool) -> GeoComputeResult<PointArray> {
        let dim = if z {
            GeoDimension::XYZ
        } else {
            GeoDimension::XY
        };
        self.overlaid(TopologicalDimension::Points, false)?
            .into_point_array(dim)
    }

    pub fn into_multi_points(self, z: bool) -> GeoComputeResult<MultiPointArray> {
        let overlaid = self.overlaid(TopologicalDimension::Points, true)?;
        with_const_flags!(Z = z; overlaid.into_multi_point_array::<Z>(self.on_row_error))
    }

    /// A row with several linestrings fails with [`GeoComputeError::Split`].
    pub fn into_line_strings(self, z: bool) -> GeoComputeResult<LineStringArray> {
        let overlaid = self.overlaid(TopologicalDimension::Lines, false)?;
        with_const_flags!(Z = z; overlaid.into_line_string_array::<Z>(self.on_row_error))
    }

    pub fn into_multi_line_strings(self, z: bool) -> GeoComputeResult<MultiLineStringArray> {
        let overlaid = self.overlaid(TopologicalDimension::Lines, true)?;
        with_const_flags!(Z = z; overlaid.into_multi_line_string_array::<Z>(self.on_row_error))
    }

    /// A row with several polygons fails with [`GeoComputeError::Split`].
    pub fn into_polygons(
        self,
        z: bool,
        new_ring_orientation: NewRingOrientation,
        new_holes: NewHoles,
    ) -> GeoComputeResult<PolygonArray> {
        self.overlaid(TopologicalDimension::Polygons, false)?
            .into_polygon_array(z, self.on_row_error, new_ring_orientation, new_holes)
    }

    /// See [`OverlayPairs::into_polygons`].
    pub fn into_multi_polygons(
        self,
        z: bool,
        new_ring_orientation: NewRingOrientation,
        new_holes: NewHoles,
    ) -> GeoComputeResult<MultiPolygonArray> {
        self.overlaid(TopologicalDimension::Polygons, true)?
            .into_multi_polygon_array(z, self.on_row_error, new_ring_orientation, new_holes)
    }

    /// Overlays every pair, and narrows each overlay down with [`kept_parts`].
    fn overlaid(
        &self,
        kept_dimension: TopologicalDimension,
        multi: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        let mut geometries = GeosRows::with_capacity(self.pairs.len());
        for overlay in self.pairs.evaluate(&self.operation, self.on_row_error) {
            let narrowed = overlay?.map(|overlay| narrowed(overlay, kept_dimension, multi));
            geometries.push(row_or_null(narrowed.transpose(), self.on_row_error)?);
        }
        Ok(GeosGeometries {
            geometries,
            size_hint: summed_size_hint(self.pairs.size_hints()),
            // GEOS builds every ring of an overlay anew.
            // The copy of `a` that a difference gives for an empty `b` has `maybe_wrong_orientation` instead.
            orientation: GeometryOrientation::GEOS_BUILT,
        })
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

fn xy_is_finite_or_error(geometry: &impl Geom) -> GeoComputeResult<()> {
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
            xy_is_finite_or_error(&geometry.get_exterior_ring()?)?;
            for n in 0..geometry.get_num_interior_rings()? {
                xy_is_finite_or_error(&geometry.get_interior_ring_n(n)?)?;
            }
            Ok(())
        }
        _ => {
            for n in 0..geometry.get_num_geometries()? {
                xy_is_finite_or_error(&geometry.get_geometry_n(n)?)?;
            }
            Ok(())
        }
    }
}

/// [kept_parts] of a row, which keeps its `maybe_wrong_orientation`.
#[inline(always)]
fn narrowed(
    overlay: GeosRow,
    kept_dimension: TopologicalDimension,
    multi: bool,
) -> GeoComputeResult<GeosRow> {
    match overlay {
        GeosRow::Empty => Ok(GeosRow::Empty),
        GeosRow::Geometry {
            geometry,
            maybe_wrong_orientation,
        } => kept_parts(geometry, maybe_wrong_orientation, kept_dimension, multi),
    }
}

/// Narrows an overlay down to its parts of `kept_dimension`,
/// as one geometry for the `into_*_array` of that kind.
/// Several parts fail with [`GeoComputeError::Split`], unless `multi=true` puts them in a multi geometry.
fn kept_parts(
    overlay: Geometry,
    maybe_wrong_orientation: bool,
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
        return Ok(GeosRow::Geometry {
            geometry: overlay,
            maybe_wrong_orientation,
        });
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
        _ if multi => Ok(GeosRow::Geometry {
            geometry: kept_dimension.create_multi(kept)?,
            maybe_wrong_orientation,
        }),
        1 => Ok(GeosRow::Geometry {
            geometry: kept.pop().expect("one part"),
            maybe_wrong_orientation,
        }),
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
