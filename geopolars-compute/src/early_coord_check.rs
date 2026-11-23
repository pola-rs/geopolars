const_enum!(
    /// Which coordinates a kernel looks for in a check before its work.
    /// A GEOS kernel sends a row holding one to `on_row_error`, a native one gives it NaN.
    /// Implements `early_coord_check`, which `geopolars/docs/source/reference/common_arguments.rst` documents.
    pub enum EarlyCoordCheck {
        Off,
        Finite,
        FiniteAndEmpty,
    }
);

#[cfg(any(feature = "native", feature = "geos"))]
#[inline(always)]
pub(crate) fn is_geoarrow_empty_point(x: f64, y: f64) -> bool {
    x.is_nan() && y.is_nan()
}

#[cfg(any(feature = "native", feature = "geos"))]
#[inline(always)]
fn xy_passes<const EARLY_COORD_CHECK: u8>(x: f64, y: f64) -> bool {
    match const { EarlyCoordCheck::from_u8(EARLY_COORD_CHECK) } {
        EarlyCoordCheck::Off => true,
        EarlyCoordCheck::Finite => (x.is_finite() & y.is_finite()) | is_geoarrow_empty_point(x, y),
        EarlyCoordCheck::FiniteAndEmpty => x.is_finite() & y.is_finite(),
    }
}

/// Whether every `x` and `y` passes `EARLY_COORD_CHECK`. z and m are not checked.
#[cfg(any(feature = "native", feature = "geos"))]
#[inline(always)]
pub(crate) fn every_xy_passes<const EARLY_COORD_CHECK: u8>(x: &[f64], y: &[f64]) -> bool {
    // Does not stop at the first that fails, so that it vectorises.
    x.iter().zip(y).fold(true, |ok, (&x, &y)| {
        ok & xy_passes::<EARLY_COORD_CHECK>(x, y)
    })
}

/// The first `x` and `y` that fail `EARLY_COORD_CHECK`, if any.
#[cfg(feature = "geos")]
#[inline(always)]
pub(crate) fn first_failing_xy<const EARLY_COORD_CHECK: u8>(
    x: &[f64],
    y: &[f64],
) -> Option<(f64, f64)> {
    if every_xy_passes::<EARLY_COORD_CHECK>(x, y) {
        return None;
    }
    let failing = x
        .iter()
        .zip(y)
        .map(|(&x, &y)| (x, y))
        .find(|&(x, y)| !xy_passes::<EARLY_COORD_CHECK>(x, y))
        .expect("a failing coordinate, as the check above failed");
    Some(failing)
}
