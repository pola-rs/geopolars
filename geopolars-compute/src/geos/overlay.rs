//! The overlays of two geometries (their intersection or difference), forwarded to GEOS.
//! See `intersection` and `difference` in Python for the rules.

use geopolars_core::output::SizeHint;
use geopolars_core::{
    GeoDimension, GeometryArray, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use geos::{CoordType, Geom, Geometry, GeometryTypes};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};

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

    /// Overlays one pair of rows from [`Overlay::pairs`], which decides the rows it sees as `Empty`.
    #[inline(always)]
    fn row(
        self,
        a: &GeosRow,
        b: &GeosRow,
        kept_dimension: TopologicalDimension,
        multi: bool,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<OverlaidRow> {
        let ignore = ignore_per_row_errors;
        match self {
            Self::Intersection => Ok(OverlaidRow {
                geometry: intersect_rows(a, b, kept_dimension, multi, ignore)?,
                is_copy_of_a: false,
            }),
            Self::Difference => {
                difference_rows(a, b, kept_dimension, multi, ignore, early_finite_check)
            }
        }
    }
}

/// Two arrays of geometries converted to GEOS, with their rows paired up to overlay.
/// A single row on either side is paired with every row of the other.
/// Each `into_*` method narrows every overlay down to its parts of one dimension,
/// and copies them into our array of that kind.
/// m is always dropped.
pub struct OverlayPairs {
    pairs: GeosPairs,
    overlay: Overlay,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

struct Overlaid {
    geometries: GeosGeometries,
    /// The rows that are a copy of `a`, or `None` if there are none.
    kept_as_is: Option<Bitmap>,
}

impl OverlayPairs {
    /// Implements the Python `intersection` and `difference`, which document their behaviour.
    /// Each side is converted to GEOS in full,
    /// except for the rows whose overlay an empty partner already makes empty.
    /// For a difference, those are only the rows of `b` paired with an empty `a`.
    /// So a row paired with a null can still fail, as can a row of `a` paired with an empty `b`.
    /// A row that fails here or in the `into_*` method is null if `ignore_per_row_errors=true`.
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

    /// Several points fail with [`GeoComputeError::Split`].
    pub fn into_points(self, z: bool) -> GeoComputeResult<PointArray> {
        let dim = if z {
            GeoDimension::XYZ
        } else {
            GeoDimension::XY
        };
        self.overlaid(TopologicalDimension::Points, false)?
            .geometries
            .into_point_array(dim)
    }

    pub fn into_multi_points(self, z: bool) -> GeoComputeResult<MultiPointArray> {
        let overlaid = self.overlaid(TopologicalDimension::Points, true)?;
        with_const_flags!(Z = z; overlaid.geometries.into_multi_point_array::<Z>(self.ignore_per_row_errors))
    }

    /// Several linestrings fail with [`GeoComputeError::Split`].
    pub fn into_line_strings(self, z: bool) -> GeoComputeResult<LineStringArray> {
        let overlaid = self.overlaid(TopologicalDimension::Lines, false)?;
        with_const_flags!(Z = z; overlaid.geometries.into_line_string_array::<Z>(self.ignore_per_row_errors))
    }

    pub fn into_multi_line_strings(self, z: bool) -> GeoComputeResult<MultiLineStringArray> {
        let overlaid = self.overlaid(TopologicalDimension::Lines, true)?;
        with_const_flags!(Z = z; overlaid.geometries.into_multi_line_string_array::<Z>(self.ignore_per_row_errors))
    }

    /// Several polygons fail with [`GeoComputeError::Split`].
    /// GEOS gives a clockwise exterior, so `ccw_outer=true` reverses every ring.
    /// A row of a difference with an empty `b` is a copy of `a`, so its rings are kept as they are.
    pub fn into_polygons(self, z: bool, ccw_outer: bool) -> GeoComputeResult<PolygonArray> {
        let overlaid = self.overlaid(TopologicalDimension::Polygons, false)?;
        overlaid.geometries.into_polygon_array(
            z,
            self.ignore_per_row_errors,
            ccw_outer,
            overlaid.kept_as_is.as_ref(),
        )
    }

    /// See [`OverlayPairs::into_polygons`] for `ccw_outer`.
    pub fn into_multi_polygons(
        self,
        z: bool,
        ccw_outer: bool,
    ) -> GeoComputeResult<MultiPolygonArray> {
        let overlaid = self.overlaid(TopologicalDimension::Polygons, true)?;
        overlaid.geometries.into_multi_polygon_array(
            z,
            self.ignore_per_row_errors,
            ccw_outer,
            overlaid.kept_as_is.as_ref(),
        )
    }

    /// Overlays every pair, and narrows each overlay down with [`kept_parts`].
    fn overlaid(
        &self,
        kept_dimension: TopologicalDimension,
        multi: bool,
    ) -> GeoComputeResult<Overlaid> {
        let (ignore, early) = (self.ignore_per_row_errors, self.early_finite_check);
        let mut kept_as_is = MutableBitmap::new();
        let geometries = self
            .pairs
            .iter()
            .map(|(a, b)| {
                let row = self
                    .overlay
                    .row(a, b, kept_dimension, multi, ignore, early)?;
                kept_as_is.push(row.is_copy_of_a);
                Ok(row.geometry)
            })
            .collect::<GeoComputeResult<Vec<_>>>()?;
        let kept_as_is: Bitmap = kept_as_is.into();
        let kept_as_is = (kept_as_is.set_bits() > 0).then_some(kept_as_is);
        Ok(Overlaid {
            geometries: GeosGeometries {
                geometries,
                size_hint: summed_size_hint(self.pairs.size_hints()),
            },
            kept_as_is,
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

struct OverlaidRow {
    geometry: GeosRow,
    is_copy_of_a: bool,
}

#[inline(always)]
fn difference_rows(
    a: &GeosRow,
    b: &GeosRow,
    kept_dimension: TopologicalDimension,
    multi: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<OverlaidRow> {
    let (difference, is_copy_of_a) = match (a, b) {
        (GeosRow::Null, _) | (_, GeosRow::Null) => (Ok(GeosRow::Null), false),
        (GeosRow::Empty, _) => (Ok(GeosRow::Empty), false),
        // Here `b` is empty itself, as the only other reason it is taken as empty is an empty `a`,
        // which the arm above handles (see `GeosPairs::new_skipping_partners_of_empty_a`).
        // Nothing is taken away, so `a` is stored as it is, without computing a difference.
        // Its own kind fits the output, as the `into_multi_*` methods take a single geometry as one part.
        // It is cloned, as the pairs only lend their rows, and a single `a` is shared by every row.
        // With `early_finite_check=true`, converting `a` has already refused a non-finite x or y.
        // The copy refuses it with `early_finite_check=false` too, so that the option never changes a copied row.
        // GEOS itself does not fail on every such vertex
        // (a line minus a polygon it does not touch keeps a NaN),
        // so a copied row can be stricter than one GEOS computes.
        (GeosRow::Geometry(a), GeosRow::Empty) => {
            let finite = if early_finite_check {
                Ok(())
            } else {
                difference_rows_xy_is_finite_or_error(a)
            };
            let copy = finite.and_then(|()| Ok(GeosRow::Geometry(Geom::clone(a)?)));
            (copy, true)
        }
        // TODO: Also give `a` as it is when `b` does not touch it, as for an empty `b`.
        // GEOS now re-orients and nodes `a` even then, so it can differ from a copy, and even split.
        (GeosRow::Geometry(a), GeosRow::Geometry(b)) => (
            a.difference(b)
                .map_err(GeoComputeError::from)
                .and_then(|difference| kept_parts(difference, kept_dimension, multi)),
            false,
        ),
    };
    let geometry =
        row_or_null(difference.map(Some), ignore_per_row_errors)?.unwrap_or(GeosRow::Null);
    Ok(OverlaidRow {
        geometry,
        is_copy_of_a,
    })
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
/// in a single geometry for the `into_*_array` of that kind.
/// With `multi=true`, several parts become a multi geometry instead of failing with [`GeoComputeError::Split`].
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
    // A collection can mix dimensions.
    // A multi geometry has to be unpacked into a single one, or refused if it has several parts.
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
