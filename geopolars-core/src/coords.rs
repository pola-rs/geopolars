//! Views over the coordinate struct of a geometry column.

use std::ops::Range;

use polars_arrow::array::{Array, PrimitiveArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_core::prelude::PolarsResult;

use crate::output::{BufferAxes, OwnedAxes};
use crate::{downcast, GeoDimension};

/// The fewest vertices a ring can have: a triangle, plus the vertex closing it.
pub const MIN_RING_VERTICES: usize = 4;

/// As `pl.Expr.is_close`'s `abs_tol` and `rel_tol`.
/// The default, both `0.0`, only lets equal values be close.
#[derive(Debug, Default, Clone, Copy)]
pub struct Tolerance {
    pub abs_tol: f64,
    pub rel_tol: f64,
}

/// Whether two coordinate values are almost the same, as `pl.Expr.is_close`:
/// `|a - b| <= max(rel_tol * max(|a|, |b|), abs_tol)`.
/// Infinities are only close to themselves, and NaN is close to nothing.
#[inline(always)]
pub fn is_close(a: f64, b: f64, tolerance: Tolerance) -> bool {
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= (tolerance.rel_tol * a.abs().max(b.abs())).max(tolerance.abs_tol)
}

/// [is_close] with `rel_tol=0.0`, for a finite `abs_tol`.
#[inline(always)]
pub fn is_close_abs(a: f64, b: f64, abs_tol: f64) -> bool {
    // An infinity is close to itself through `a == b`, as `a - b` is NaN.
    a == b || (a - b).abs() <= abs_tol
}

/// Whether two [positions](ValidCoords::position) are equal on every axis.
#[inline(always)]
pub fn same_position(a: [f64; 3], b: [f64; 3]) -> bool {
    a == b
}

/// Whether coordinates `i` and `j` of `coords` lie at exactly the same place:
/// equal in x and y, and in z if `Z=true`.
/// m is a measure rather than a position, so it is not compared.
#[inline(always)]
pub fn same_place<const Z: bool>(coords: &ValidCoords, i: usize, j: usize) -> bool {
    coords.x()[i] == coords.x()[j]
        && coords.y()[i] == coords.y()[j]
        && (!Z || {
            let z = coords.z().expect("a z, with `Z`");
            z[i] == z[j]
        })
}

/// [same_place], but every axis only has to be [`is_close`] within `tolerance`.
#[inline(always)]
pub fn same_place_within<const Z: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
    tolerance: Tolerance,
) -> bool {
    is_close(coords.x()[i], coords.x()[j], tolerance)
        && is_close(coords.y()[i], coords.y()[j], tolerance)
        && (!Z || {
            let z = coords.z().expect("a z, with `Z`");
            is_close(z[i], z[j], tolerance)
        })
}

/// [same_place_within], with [is_close_abs] on every axis.
#[inline(always)]
pub fn same_place_within_abs<const Z: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
    abs_tol: f64,
) -> bool {
    is_close_abs(coords.x()[i], coords.x()[j], abs_tol)
        && is_close_abs(coords.y()[i], coords.y()[j], abs_tol)
        && (!Z || {
            let z = coords.z().expect("a z, with `Z`");
            is_close_abs(z[i], z[j], abs_tol)
        })
}

/// Whether the coordinates `vertices` of `coords` make a ring:
/// at least [MIN_RING_VERTICES] of them, with the last at the [same place](same_place) as the first.
/// See [same_place] for `Z`.
#[inline(always)]
pub fn is_ring<const Z: bool>(coords: &ValidCoords, vertices: Range<usize>) -> bool {
    vertices.len() >= MIN_RING_VERTICES && same_place::<Z>(coords, vertices.start, vertices.end - 1)
}

/// Coordinates that are all assumed to be valid.
/// Use for coordinates that **cannot** include null coordinates (nested geometries).
/// Otherwise, use [Coords].
///
/// See the README on views.
#[derive(Clone)]
pub struct ValidCoords {
    x: Buffer<f64>,
    y: Buffer<f64>,
    /// `None` for a dimension without a `z`.
    z: Option<Buffer<f64>>,
    /// `None` for a dimension without an `m`.
    m: Option<Buffer<f64>>,
}

impl ValidCoords {
    pub fn new(coords: &dyn Array) -> PolarsResult<Self> {
        let coords: &StructArray = downcast(coords, "a coordinate struct")?;
        let arrays = AxisArrays::new(coords)?;
        Ok(Self {
            x: arrays.x.values().clone(),
            y: arrays.y.values().clone(),
            z: arrays.z.map(|z| z.values().clone()),
            m: arrays.m.map(|m| m.values().clone()),
        })
    }

    pub fn from_axes((x, y, z, m): OwnedAxes) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z: z.map(Buffer::from),
            m: m.map(Buffer::from),
        }
    }

    /// These coordinates, with x, y and z replaced where they are `Some`.
    /// The other axes share their buffers.
    /// A `Some` z needs a dimension with a `z`.
    pub fn with_positions(&self, [x, y, z]: [Option<Vec<f64>>; 3]) -> Self {
        assert!(
            z.is_none() || self.z.is_some(),
            "a z for coordinates without one"
        );
        Self {
            x: x.map_or_else(|| self.x.clone(), Buffer::from),
            y: y.map_or_else(|| self.y.clone(), Buffer::from),
            z: z.map(Buffer::from).or_else(|| self.z.clone()),
            m: self.m.clone(),
        }
    }

    /// These coordinates copied into new buffers, with each of `ranges` reversed.
    /// `ranges` is run once per axis.
    pub fn with_ranges_reversed(&self, ranges: impl Iterator<Item = Range<usize>> + Clone) -> Self {
        let reversed = |axis: &[f64]| {
            let mut axis = axis.to_vec();
            for range in ranges.clone() {
                axis[range].reverse();
            }
            axis
        };
        Self::from_axes((
            reversed(&self.x),
            reversed(&self.y),
            self.z.as_deref().map(&reversed),
            self.m.as_deref().map(&reversed),
        ))
    }

    /// The coordinates at `ranges`, in order, copied into new buffers in one pass.
    /// `ranges` is run twice: once to size the buffers, once to copy.
    /// An axis these coordinates don't have is not copied either.
    pub fn copy_ranges(&self, ranges: impl Iterator<Item = Range<usize>> + Clone) -> Self {
        match (self.z.is_some(), self.m.is_some()) {
            (false, false) => self.copy_ranges_with::<false, false>(ranges),
            (true, false) => self.copy_ranges_with::<true, false>(ranges),
            (false, true) => self.copy_ranges_with::<false, true>(ranges),
            (true, true) => self.copy_ranges_with::<true, true>(ranges),
        }
    }

    /// [copy_ranges](Self::copy_ranges), copying z only if `Z=true`, and m only if `M=true`.
    #[inline(always)]
    fn copy_ranges_with<const Z: bool, const M: bool>(
        &self,
        ranges: impl Iterator<Item = Range<usize>> + Clone,
    ) -> Self {
        let len = ranges.clone().map(|range| range.len()).sum();
        let (src_z, src_m) = (
            self.z.as_deref().filter(|_| Z).unwrap_or_default(),
            self.m.as_deref().filter(|_| M).unwrap_or_default(),
        );
        let (mut x, mut y) = (Vec::with_capacity(len), Vec::with_capacity(len));
        let mut z = Vec::with_capacity(if Z { len } else { 0 });
        let mut m = Vec::with_capacity(if M { len } else { 0 });
        for range in ranges {
            x.extend_from_slice(&self.x[range.clone()]);
            y.extend_from_slice(&self.y[range.clone()]);
            if Z {
                z.extend_from_slice(&src_z[range.clone()]);
            }
            if M {
                m.extend_from_slice(&src_m[range]);
            }
        }
        Self::from_axes((x, y, Z.then_some(z), M.then_some(m)))
    }

    pub(crate) fn into_axes(self) -> BufferAxes {
        (self.x, self.y, self.z, self.m)
    }

    /// How many coordinates there are.
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Coordinates `range` of these, sharing their buffers.
    pub(crate) fn sliced(&self, range: Range<usize>) -> Self {
        let sliced = |axis: &Buffer<f64>| axis.clone().sliced(range.clone());
        Self {
            x: sliced(&self.x),
            y: sliced(&self.y),
            z: self.z.as_ref().map(sliced),
            m: self.m.as_ref().map(sliced),
        }
    }

    /// The position of coordinate `i`: x, y, and with `Z` its z (`0.0` without).
    #[inline(always)]
    pub fn position<const Z: bool>(&self, i: usize) -> [f64; 3] {
        [
            self.x[i],
            self.y[i],
            if Z { self.z.as_ref().unwrap()[i] } else { 0.0 },
        ]
    }

    #[inline(always)]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    #[inline(always)]
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// `None` for a dimension without a `z`.
    #[inline(always)]
    pub fn z(&self) -> Option<&[f64]> {
        self.z.as_deref()
    }

    /// `None` for a dimension without an `m`.
    #[inline(always)]
    pub fn m(&self) -> Option<&[f64]> {
        self.m.as_deref()
    }

    /// Every value of axis `n`, in storage order: x, y, then z and/or m.
    pub fn axis(&self, n: usize) -> &[f64] {
        let values = match n {
            0 => Some(&self.x),
            1 => Some(&self.y),
            2 => self.z.as_ref().or(self.m.as_ref()),
            3 => self.m.as_ref(),
            _ => None,
        };
        values.unwrap_or_else(|| panic!("coordinate index {n} out of range"))
    }

    /// Value `n` of coordinate `i`, in the order of [`axis`](Self::axis).
    #[inline]
    pub fn nth(&self, i: usize, n: usize) -> f64 {
        self.axis(n)[i]
    }

    /// The dimension these coordinates carry.
    pub fn dim(&self) -> GeoDimension {
        match (self.z.is_some(), self.m.is_some()) {
            (false, false) => GeoDimension::XY,
            (true, false) => GeoDimension::XYZ,
            (false, true) => GeoDimension::XYM,
            (true, true) => GeoDimension::XYZM,
        }
    }
}

/// Coordinate data that might hold nulls.
/// Considers validity of individual components of the coordinates
/// ('x' may be valid, but 'y' may not be).
/// Use this in validation only.
pub struct AxisArrays<'a> {
    pub x: &'a PrimitiveArray<f64>,
    pub y: &'a PrimitiveArray<f64>,
    /// `None` for a dimension without a `z`.
    pub z: Option<&'a PrimitiveArray<f64>>,
    /// `None` for a dimension without an `m`.
    pub m: Option<&'a PrimitiveArray<f64>>,
}

impl<'a> AxisArrays<'a> {
    pub fn new(coords: &'a StructArray) -> PolarsResult<Self> {
        // The dtype fixes the field order, so x and y always come first.
        let x = downcast(coords.values()[0].as_ref(), "f64 `x` coordinates")?;
        let y = downcast(coords.values()[1].as_ref(), "f64 `y` coordinates")?;
        let field = |name: &str| {
            coords
                .fields()
                .iter()
                .position(|field| field.name.as_str() == name)
                .map(|i| downcast(coords.values()[i].as_ref(), "f64 coordinates"))
                .transpose()
        };
        Ok(Self {
            x,
            y,
            z: field("z")?,
            m: field("m")?,
        })
    }
}

/// Coordinates where individual coordinates can be null.
/// Use for coordinates that **can** include null coordinates (columns of Points).
/// Otherwise, use [ValidCoords].
///
/// See the README on views.
pub struct Coords {
    /// Which coordinates are there.
    validity: Option<Bitmap>,
    axes: ValidCoords,
}

impl Coords {
    pub fn new(coords: &dyn Array) -> PolarsResult<Self> {
        Ok(Self {
            validity: coords.validity().cloned(),
            axes: ValidCoords::new(coords)?,
        })
    }

    pub fn from_valid(axes: ValidCoords, validity: Option<Bitmap>) -> Self {
        Self { validity, axes }
    }

    pub(crate) fn into_parts(self) -> (BufferAxes, Option<Bitmap>) {
        (self.axes.into_axes(), self.validity)
    }

    #[inline(always)]
    pub fn is_valid(&self, i: usize) -> bool {
        self.validity
            .as_ref()
            .is_none_or(|validity| validity.get_bit(i))
    }

    /// The coordinates, for once it is known which are there.
    #[inline]
    pub fn ignoring_validity(&self) -> &ValidCoords {
        &self.axes
    }

    /// `None` when every coordinate is there.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Whether every coordinate is there.
    #[inline]
    pub fn is_all_valid(&self) -> bool {
        self.validity
            .as_ref()
            .is_none_or(|validity| validity.unset_bits() == 0)
    }

    pub fn xy(&self, i: usize) -> Option<(f64, f64)> {
        self.xy_checking_nulls::<true>(i)
    }

    /// [`xy`](Self::xy), checking whether it is null only with `NULLS`.
    /// Without, it assumes every coordinate is there.
    #[inline(always)]
    pub fn xy_checking_nulls<const NULLS: bool>(&self, i: usize) -> Option<(f64, f64)> {
        (!NULLS || self.is_valid(i)).then(|| (self.axes.x[i], self.axes.y[i]))
    }

    /// The position of coordinate `i`: x, y, and with `Z` its z (`0.0` without).
    /// With `NULLS`, `None` if it is null.
    /// Without, it assumes every coordinate is there.
    #[inline(always)]
    pub fn position<const NULLS: bool, const Z: bool>(&self, i: usize) -> Option<[f64; 3]> {
        (!NULLS || self.is_valid(i)).then(|| self.axes.position::<Z>(i))
    }

    /// See [`ValidCoords::len`].
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.axes.len()
    }

    /// See [`ValidCoords::sliced`].
    pub(crate) fn sliced(&self, range: Range<usize>) -> Self {
        let validity =
            (self.validity.clone()).map(|validity| validity.sliced(range.start, range.len()));
        Self {
            validity,
            axes: self.axes.sliced(range),
        }
    }

    /// See [`ValidCoords::nth`].
    #[inline]
    pub fn nth(&self, i: usize, n: usize) -> f64 {
        self.axes.nth(i, n)
    }

    /// See [`ValidCoords::dim`].
    #[inline]
    pub fn dim(&self) -> GeoDimension {
        self.axes.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_close_abs_matches_is_close_without_rel_tol() {
        let values = [
            0.0,
            -0.0,
            1.0,
            1.0 + f64::EPSILON,
            -1.0,
            1e300,
            -1e300,
            f64::MIN_POSITIVE / 2.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for abs_tol in [0.0, 1e-9, 1.0, -1.0, f64::NAN, f64::MAX] {
            let tolerance = Tolerance {
                abs_tol,
                rel_tol: 0.0,
            };
            for a in values {
                for b in values {
                    assert_eq!(
                        is_close_abs(a, b, abs_tol),
                        is_close(a, b, tolerance),
                        "a={a}, b={b}, abs_tol={abs_tol}"
                    );
                }
            }
        }
    }
}
