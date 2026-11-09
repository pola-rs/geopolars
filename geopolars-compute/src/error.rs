use std::fmt;

use geopolars_core::MIN_RING_VERTICES;
use proj::ProjError;

/// An error from one of the kernels of this crate.
#[derive(Debug)]
pub enum GeoComputeError {
    /// A coordinate PROJ could not reproject.
    Reproject {
        position: (f64, f64),
        source: ProjError,
    },
    /// A transformation PROJ could not create.
    Transformation {
        from: String,
        to: String,
        message: String,
    },
    /// A linestring that has to be a ring, but is not one.
    NotARing { vertices: usize },
    /// A ring that has to be closed or empty, but is neither.
    NotClosed { vertices: usize },
    /// A coordinate with a NaN or infinite x or y, which GEOS cannot take.
    NotFinite { position: (f64, f64) },
    /// A coordinate with a NaN or infinite value that the kernel cannot skip.
    /// `z` and `m` are `None` where the coordinate does not have them.
    NotFiniteCoordinate {
        x: f64,
        y: f64,
        z: Option<f64>,
        m: Option<f64>,
    },
    /// A NaN or infinite distance.
    NotFiniteDistance { distance: f64 },
    /// A negative distance for a buffer of one side, whose side is not given by the sign.
    NegativeOneSidedDistance { distance: f64 },
    /// A result in several parts, where only one fits.
    Split { parts: usize },
    #[cfg(feature = "geos")]
    Geos(geos::Error),
}

pub type GeoComputeResult<T> = Result<T, GeoComputeError>;

/// The result of one row, or `None` if it failed and `ignore_per_row_errors=true`.
#[cfg(any(feature = "geos", feature = "rsgeo"))]
#[inline(always)]
pub(crate) fn row_or_null<T>(
    result: GeoComputeResult<Option<T>>,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<Option<T>> {
    match result {
        Err(_) if ignore_per_row_errors => Ok(None),
        result => result,
    }
}

impl fmt::Display for GeoComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reproject {
                position: (x, y),
                source,
            } => write!(f, "failed to reproject ({x}, {y}): {source}"),
            Self::Transformation { from, to, message } => {
                write!(f, "cannot reproject from {from} to {to}: {message}")
            }
            Self::NotARing { vertices } => write!(
                f,
                "expects rings (at least {MIN_RING_VERTICES} vertices, the last at the first), \
                got a linestring of {vertices} vertices that is not one"
            ),
            Self::NotClosed { vertices } => write!(
                f,
                "expects closed rings (the last vertex at the first) or empty ones, \
                got a ring of {vertices} vertices that is not closed"
            ),
            Self::NotFinite { position: (x, y) } => write!(
                f,
                "GEOS expects finite x and y, got ({x}, {y}). \
                GEOS does not support empty points inside a geometry"
            ),
            Self::NotFiniteCoordinate { x, y, z, m } => {
                write!(f, "expects finite coordinates, got ({x}, {y}")?;
                for value in [z, m].into_iter().flatten() {
                    write!(f, ", {value}")?;
                }
                write!(f, ")")
            }
            Self::NotFiniteDistance { distance } => {
                write!(f, "expects a finite distance, got: {distance}")
            }
            Self::NegativeOneSidedDistance { distance } => write!(
                f,
                "a buffer of one side expects a distance of at least 0, got: {distance}"
            ),
            Self::Split { parts } => write!(
                f,
                "the result split into {parts} parts, which a single geometry cannot hold; \
                set `allow_splitting=True` to get a multi kind"
            ),
            #[cfg(feature = "geos")]
            Self::Geos(source) => write!(f, "{source}"),
        }
    }
}

impl std::error::Error for GeoComputeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reproject { source, .. } => Some(source),
            Self::Transformation { .. }
            | Self::NotARing { .. }
            | Self::NotClosed { .. }
            | Self::NotFinite { .. }
            | Self::NotFiniteCoordinate { .. }
            | Self::NotFiniteDistance { .. }
            | Self::NegativeOneSidedDistance { .. }
            | Self::Split { .. } => None,
            #[cfg(feature = "geos")]
            Self::Geos(source) => Some(source),
        }
    }
}

#[cfg(feature = "geos")]
impl From<geos::Error> for GeoComputeError {
    fn from(source: geos::Error) -> Self {
        Self::Geos(source)
    }
}
