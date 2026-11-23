const_enum!(
    /// Which coordinates a kernel rejects in a pass before its work.
    /// A row with such a coordinate goes to `on_row_error`.
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

/// The first `x` and `y` that fail `EARLY_COORD_CHECK`, if any.
/// z and m are not checked.
#[cfg(any(feature = "native", feature = "geos"))]
#[inline(always)]
pub(crate) fn first_failing_xy<const EARLY_COORD_CHECK: u8>(
    x: &[f64],
    y: &[f64],
) -> Option<(f64, f64)> {
    let passes = |x: f64, y: f64| match const { EarlyCoordCheck::from_u8(EARLY_COORD_CHECK) } {
        EarlyCoordCheck::Off => true,
        EarlyCoordCheck::Finite => (x.is_finite() & y.is_finite()) | is_geoarrow_empty_point(x, y),
        EarlyCoordCheck::FiniteAndEmpty => x.is_finite() & y.is_finite(),
    };
    // Does not stop at the first that fails, so that it vectorises.
    let every_passes = x.iter().zip(y).fold(true, |ok, (&x, &y)| ok & passes(x, y));
    if every_passes {
        return None;
    }
    let failing = x
        .iter()
        .zip(y)
        .map(|(&x, &y)| (x, y))
        .find(|&(x, y)| !passes(x, y))
        .expect("a failing coordinate, as the check above failed");
    Some(failing)
}
