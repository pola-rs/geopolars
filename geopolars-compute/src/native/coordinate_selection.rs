//! The rules for single coordinates.
//! They decide which coordinate is the empty point, which is not finite, and which lies outside the bounds,
//! as the Python `filter_coordinates` and `fix_out_of_bounds` document.
//!
//! [selected_coords] marks the coordinates a [CoordinateSelection] selects, in one pass.
//! [super::selected_coords](mod@super::selected_coords) acts on them.
//! The validity checks reject the rows holding one ([which_are_not_selected]).

use geopolars_views::ValidCoords;
use polars_arrow::bitmap::Bitmap;

/// Where `x` and `y` may lie, as `[xmin, ymin, xmax, ymax]`, ends included.
pub type Bounds = [f64; 4];

/// Which coordinates to select for not being finite, or for being empty.
/// See `NonFinite` in Python, without its `action` and `too_short`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NonFinite {
    pub select_non_finite: bool,
    pub select_empty: bool,
    pub ignore_z: bool,
    pub ignore_m: bool,
}

/// Which coordinates to select.
/// A coordinate is selected if either check selects it.
#[derive(Debug, Clone, Copy, Default)]
pub struct CoordinateSelection {
    pub non_finite: Option<NonFinite>,
    /// The bounds outside which a coordinate is selected.
    /// The empty point is never selected for lying outside them.
    pub bounds: Option<Bounds>,
}

impl CoordinateSelection {
    pub fn selects_anything(&self) -> bool {
        self.non_finite.is_some() || self.bounds.is_some()
    }
}

/// Whether a coordinate is NaN on every axis it has, which makes it the empty point.
/// Pass `None` for an axis that is not there.
#[inline(always)]
fn is_empty_value(x: f64, y: f64, z: Option<f64>, m: Option<f64>) -> bool {
    x.is_nan() & y.is_nan() & z.is_none_or(f64::is_nan) & m.is_none_or(f64::is_nan)
}

/// [is_empty_value] for coordinate `i`.
#[inline(always)]
pub(crate) fn is_empty_point(coords: &ValidCoords, i: usize) -> bool {
    let (z, m) = (coords.z().map(|z| z[i]), coords.m().map(|m| m[i]));
    is_empty_value(coords.x()[i], coords.y()[i], z, m)
}

/// Whether a coordinate has a NaN or infinite value.
/// Pass `None` for an axis that is not there, or that is ignored.
#[inline(always)]
pub(crate) fn is_non_finite_value(x: f64, y: f64, z: Option<f64>, m: Option<f64>) -> bool {
    !(x.is_finite() & y.is_finite() & z.is_none_or(f64::is_finite) & m.is_none_or(f64::is_finite))
}

#[inline(always)]
fn is_within(bounds: &Bounds, x: f64, y: f64) -> bool {
    let [xmin, ymin, xmax, ymax] = *bounds;
    (xmin <= x) & (x <= xmax) & (ymin <= y) & (y <= ymax)
}

/// Packs `bits` bits into a word.
/// Bit `j` is set where `is_set(start + j)`.
#[inline(always)]
fn word_from_fn(start: usize, bits: usize, is_set: &impl Fn(usize) -> bool) -> u64 {
    (0..bits).fold(0u64, |word, j| word | (u64::from(is_set(start + j)) << j))
}

/// A bitmap of `len` bits, with bit `i` set where `is_set(i)`.
/// The bits are packed 64 at a time, which lets a branchless `is_set` compile to SIMD.
/// `may_be_set` has to hold wherever `is_set` does.
/// It is tested first, and `is_set` only for the 64 bits around one where it holds.
#[inline(always)]
fn bitmap_from_fn(
    len: usize,
    may_be_set: impl Fn(usize) -> bool,
    is_set: impl Fn(usize) -> bool,
) -> Bitmap {
    let word_of = |start: usize, bits: usize| {
        if word_from_fn(start, bits, &may_be_set) == 0 {
            0
        } else {
            word_from_fn(start, bits, &is_set)
        }
    };
    let mut bytes = Vec::with_capacity(len.div_ceil(64) * 8);
    for start in (0..len - len % 64).step_by(64) {
        bytes.extend_from_slice(&word_of(start, 64).to_le_bytes());
    }
    if !len.is_multiple_of(64) {
        bytes.extend_from_slice(&word_of(len - len % 64, len % 64).to_le_bytes());
    }
    Bitmap::from_u8_vec(bytes, len)
}

/// If `CHECK_NON_FINITE=true`, the coordinates are checked by [CoordinateSelection::non_finite].
/// If `CHECK_BOUNDS=true`, the coordinates are checked by [CoordinateSelection::bounds].
/// If `Z=true`, the z values are read too.
/// If `M=true`, the m values are read too.
///
/// The rules take values rather than [ValidCoords].
/// The pass then has no branches within a run of 64 coordinates, and compiles to SIMD.
fn selected_coords_with<
    const CHECK_NON_FINITE: bool,
    const CHECK_BOUNDS: bool,
    const Z: bool,
    const M: bool,
>(
    coords: &ValidCoords,
    filter: NonFinite,
    bounds: &Bounds,
) -> Bitmap {
    let len = coords.len();
    // Sliced to `len` up front, so that indexing them below needs no bounds checks.
    let (x, y) = (&coords.x()[..len], &coords.y()[..len]);
    let z = if Z {
        &coords.z().expect("a z, with `Z`")[..len]
    } else {
        &[]
    };
    let m = if M {
        &coords.m().expect("an m, with `M`")[..len]
    } else {
        &[]
    };
    let NonFinite {
        select_non_finite,
        select_empty,
        ignore_z,
        ignore_m,
    } = filter;
    let z_of = |i: usize| Z.then(|| z[i]);
    let m_of = |i: usize| M.then(|| m[i]);
    let is_non_finite = |i: usize| {
        let (z, m) = (z_of(i).filter(|_| !ignore_z), m_of(i).filter(|_| !ignore_m));
        CHECK_NON_FINITE & is_non_finite_value(x[i], y[i], z, m)
    };
    let is_out_of_bounds = |i: usize| CHECK_BOUNDS & !is_within(bounds, x[i], y[i]);
    // The empty point is NaN in x, so it is never finite nor within any bounds.
    // Whether a coordinate is empty therefore only matters for one that is not finite,
    // or that lies outside them.
    bitmap_from_fn(
        len,
        |i| is_non_finite(i) | is_out_of_bounds(i),
        |i| {
            let empty = is_empty_value(x[i], y[i], z_of(i), m_of(i));
            let selected_by_non_finite =
                is_non_finite(i) & ((empty & select_empty) | (!empty & select_non_finite));
            selected_by_non_finite | (is_out_of_bounds(i) & !empty)
        },
    )
}

/// Which of `coords` `selection` selects, as set bits.
/// `None` when it does not [select anything](CoordinateSelection::selects_anything).
pub fn selected_coords(coords: &ValidCoords, selection: &CoordinateSelection) -> Option<Bitmap> {
    if !selection.selects_anything() {
        return None;
    }
    let filter = selection.non_finite.unwrap_or_default();
    let bounds = selection.bounds.unwrap_or_default();
    let dim = coords.dim();
    let check_non_finite = selection.non_finite.is_some();
    let check_bounds = selection.bounds.is_some();
    Some(with_const_flags!(
        CHECK_NON_FINITE = check_non_finite,
        CHECK_BOUNDS = check_bounds,
        Z = dim.has_z(),
        M = dim.has_m();
        selected_coords_with::<CHECK_NON_FINITE, CHECK_BOUNDS, Z, M>(coords, filter, &bounds)
    ))
}

pub(crate) fn which_are_not_selected(
    coords: &ValidCoords,
    selection: &CoordinateSelection,
) -> Option<Bitmap> {
    selected_coords(coords, selection).map(|selected| !&selected)
}
