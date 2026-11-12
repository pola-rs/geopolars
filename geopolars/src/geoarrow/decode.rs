//! The geometry an encoding (`wkb`, `wkt`) is decoded into.
//! [`GeometryBuilder`](geopolars_views::GeometryBuilder) builds it.

use std::sync::Arc;

use geopolars_views::Kind;
use polars::prelude::*;
use serde::Deserialize;

use super::crs::{ExtensionMetadata, MetadataKwargs};
use super::{Geo, GeoDimension};

/// The geometry an encoding is decoded into.
/// Mirrors what `from_wkb` / `from_wkt` pass on the Python side.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeKwargs {
    /// As spelled by [`Kind::display`].
    kind: String,
    /// As spelled by [`GeoDimension::tag`].
    dimension: String,
    /// See [`MetadataKwargs::crs`].
    crs: Option<String>,
}

/// The geometry a column carrying `metadata` decodes into.
pub fn target(metadata: Arc<ExtensionMetadata>, kwargs: &DecodeKwargs) -> PolarsResult<Geo> {
    let Some(kind) = Kind::from_display(&kwargs.kind) else {
        polars_bail!(InvalidOperation: "unknown geometry: {}", kwargs.kind);
    };
    let Some(dim) = GeoDimension::from_tag(&kwargs.dimension) else {
        polars_bail!(InvalidOperation: "unknown dimension: {}", kwargs.dimension);
    };
    let metadata = MetadataKwargs {
        crs: kwargs.crs.clone(),
    }
    .apply(metadata, "geometries")?;
    Ok(Geo::new(kind, dim, metadata))
}
