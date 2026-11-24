use std::convert::Infallible;
use std::fmt;

use geopolars_views::MIN_RING_VERTICES;
use proj::ProjError;

use crate::OnRowError;

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
    GeosNotFinite { position: (f64, f64) },
    /// A coordinate with a NaN or infinite value where a native kernel needs a finite one.
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
    /// A hole with vertices for a polygon with an empty exterior.
    HoleInEmptyExterior,
    /// A coordinate at `position` outside `bounds`, the `[xmin, ymin, xmax, ymax]` of its CRS.
    OutOfBounds {
        position: (f64, f64),
        bounds: [f64; 4],
    },
    /// A geometry with vertices but without the length or area its centroid is weighted by.
    CentroidWithoutWeight {
        weighed_by: CentroidWeight,
        /// Whether `method="3d"` could measure it, as it has a z that `method="footprint"` ignores.
        suggest_3d: bool,
    },
    /// A geometry whose length or area underflows to 0 as an `f64`.
    CentroidUnderflow {
        weighed_by: CentroidWeight,
        /// How far the vertices lie from the first, at most.
        extent: f64,
    },
    /// A polygon whose holes take out more area than its exterior holds.
    HolesExceedExterior,
    #[cfg(feature = "geos")]
    Geos(geos::Error),
}

pub type GeoComputeResult<T> = Result<T, GeoComputeError>;

/// What a centroid is weighted by: the length of a linestring, or the area of a polygon.
#[derive(Clone, Copy, Debug)]
pub enum CentroidWeight {
    Length,
    Area,
}

impl CentroidWeight {
    fn name(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Area => "area",
        }
    }
}

/// The result of one row, or `None` if it failed and `on_row_error=`[OnRowError::Null].
#[cfg(any(feature = "native", feature = "geos", feature = "rsgeo"))]
#[inline(always)]
pub(crate) fn row_or_null<T, E>(
    result: Result<Option<T>, E>,
    on_row_error: OnRowError,
) -> Result<Option<T>, E> {
    match result {
        Err(_) if on_row_error == OnRowError::Null => Ok(None),
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
            Self::GeosNotFinite { position: (x, y) } => write!(
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
                set `promote_to_multi` to \"auto\" or \"always\" to get a multi kind"
            ),
            Self::HoleInEmptyExterior => write!(
                f,
                "a polygon with an empty exterior got a hole with vertices, which is not valid OGC; \
                set `empty_polygon_exterior` to \"null\", \"empty\" or \"keep\" not to raise"
            ),
            Self::OutOfBounds {
                position: (x, y),
                bounds: [xmin, ymin, xmax, ymax],
            } => write!(
                f,
                "({x}, {y}) lies outside the bounds of the CRS, \
                x from {xmin} to {xmax} and y from {ymin} to {ymax}; \
                set `on_out_of_bounds` to \"null\" or \"ignore\" not to raise"
            ),
            Self::CentroidWithoutWeight {
                weighed_by,
                suggest_3d,
            } => {
                let weight = weighed_by.name();
                let degenerate = match weighed_by {
                    CentroidWeight::Length => "every vertex on one spot",
                    CentroidWeight::Area => "every vertex on one line",
                };
                write!(
                    f,
                    "a centroid needs a geometry with {weight}, \
                    got one without (such as one with {degenerate})"
                )?;
                if *suggest_3d {
                    write!(
                        f,
                        "; a vertical geometry only has {weight} with `method=\"3d\"`"
                    )?;
                }
                Ok(())
            }
            Self::CentroidUnderflow { weighed_by, extent } => write!(
                f,
                "a centroid needs a geometry with {weight}, got one whose {weight} is too small \
                for a float (every vertex lies within {extent:e} of the first); \
                scale the geometry up first",
                weight = weighed_by.name(),
            ),
            Self::HolesExceedExterior => write!(
                f,
                "a centroid needs a polygon with area, got one whose holes take out more area \
                than its exterior holds, which is not valid OGC"
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
            | Self::GeosNotFinite { .. }
            | Self::NotFiniteCoordinate { .. }
            | Self::NotFiniteDistance { .. }
            | Self::NegativeOneSidedDistance { .. }
            | Self::Split { .. }
            | Self::HoleInEmptyExterior
            | Self::OutOfBounds { .. }
            | Self::CentroidWithoutWeight { .. }
            | Self::CentroidUnderflow { .. }
            | Self::HolesExceedExterior => None,
            #[cfg(feature = "geos")]
            Self::Geos(source) => Some(source),
        }
    }
}

impl From<Infallible> for GeoComputeError {
    fn from(never: Infallible) -> Self {
        match never {}
    }
}

#[cfg(feature = "geos")]
impl From<geos::Error> for GeoComputeError {
    fn from(source: geos::Error) -> Self {
        Self::Geos(source)
    }
}
