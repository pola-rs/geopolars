//! The GeoParquet metadata: the JSON under a Parquet file's `geo` key,
//!
//! GeoParquet keeps per column what GeoArrow keeps in a dtype's metadata,
//! but in its own format (see `docs/geoparquet-1.1.0p1.md`).
//! This module converts between our metadata representation and GeoParquet's.
//! Only GeoParquet 1.x is supported, as 2.0rc breaks with upstream

use std::collections::HashMap;
use std::sync::Arc;

use geoarrow_schema::Edges;
use geopolars_core::Kind;
use indexmap::IndexMap;
use polars::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::geoarrow::bbox::{self, GeoBox};
use crate::geoarrow::crs::ExtensionMetadata;
use crate::geoarrow::encoded::{describe_encoded, Encoded, Encoding};
use crate::geoarrow::{coord, describe, Geo, GeoDimension};

/// The GeoParquet version this crate writes.
const VERSION: &str = "1.1.0";

/// The CRS of a column without a `crs` key.
/// Follows geoparquet v1.1.0.
const DEFAULT_CRS: &str = "OGC:CRS84";

/// The value of the `geo` key in a Parquet file's metadata.
///
/// Fields we don't use (such as the file-wide `bbox`) are ignored when reading,
/// as the spec asks of readers.
#[derive(Debug, Serialize, Deserialize)]
pub struct GeoParquetMetadata {
    pub version: String,
    pub primary_column: String,
    /// Keyed by column name, in schema order.
    pub columns: IndexMap<String, GeoParquetColumn>,
}

/// How one geometry column is stored.
#[derive(Debug, Serialize, Deserialize)]
pub struct GeoParquetColumn {
    pub encoding: ColumnEncoding,
    /// `Point`, `Polygon Z`, ...; empty if not known.
    pub geometry_types: Vec<String>,
    /// A missing key (`None`) means OGC:CRS84.
    /// An explicit `null` (`Some(None)`) is an unknown CRS.
    /// Different from GeoArrow, where a missing CRS is unknown.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub crs: Option<Option<Map<String, Value>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edges: Option<ParquetEdges>,
    /// Only read so that it can be refused: we don't support dynamic CRSs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epoch: Option<f64>,
    /// A column of simpler shapes, one per geometry, to filter on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covering: Option<Covering>,
}

/// The `covering` of a geometry column.
/// The spec only knows one encoding, `bbox`.
#[derive(Debug, Serialize, Deserialize)]
pub struct Covering {
    pub bbox: BboxCovering,
}

/// Where each bound of a bbox covering is: a path of
/// `[column, field]` into a struct column, the same column for every bound.
/// z is optional, also for a geometry that has one.
#[derive(Debug, Serialize, Deserialize)]
pub struct BboxCovering {
    xmin: [String; 2],
    ymin: [String; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    zmin: Option<[String; 2]>,
    xmax: [String; 2],
    ymax: [String; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    zmax: Option<[String; 2]>,
}

/// The layout of a geometry column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnEncoding {
    Wkb,
    /// GeoArrow's separated coordinates.
    Native(Kind),
}

/// The edges GeoParquet 1.1 knows.
/// GeoArrow knows a couple more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParquetEdges {
    Planar,
    Spherical,
}

impl GeoParquetMetadata {
    /// The metadata describing every geometry column of `schema`.
    /// `None` if it has none.
    /// `coverings` is the bbox column of every geometry column that has one.
    pub fn from_schema(
        schema: &Schema,
        coverings: &HashMap<String, String>,
    ) -> PolarsResult<Option<Self>> {
        let mut columns = IndexMap::new();
        for (name, dtype) in schema.iter() {
            if is_geometry(dtype) {
                let mut column = GeoParquetColumn::from_dtype(dtype)?;
                if let Some(bbox) = coverings.get(name.as_str()) {
                    column.covering = Some(Covering::for_column(name, dtype, bbox, schema)?);
                }
                columns.insert(name.to_string(), column);
            }
        }
        if let Some(name) = coverings.keys().find(|name| !columns.contains_key(*name)) {
            polars_bail!(ColumnNotFound: "`{name}` is not a geometry column to cover");
        }
        let Some(primary_column) = columns.keys().next().cloned() else {
            return Ok(None);
        };
        Ok(Some(Self {
            version: VERSION.to_owned(),
            primary_column,
            columns,
        }))
    }

    /// Parses and validates the value of a file's `geo` key.
    pub fn parse(geo: &str) -> PolarsResult<Self> {
        let metadata: Self = serde_json::from_str(geo)
            .map_err(|e| polars_err!(ComputeError: "invalid GeoParquet metadata: {e}"))?;
        polars_ensure!(
            metadata.version.starts_with("1."),
            ComputeError: "GeoParquet {} is not supported yet, only 1.x is", metadata.version
        );
        polars_ensure!(
            metadata.columns.contains_key(&metadata.primary_column),
            ComputeError: "the GeoParquet primary column `{}` is not one of its geometry columns",
            metadata.primary_column
        );
        for (name, column) in &metadata.columns {
            polars_ensure!(
                column.epoch.is_none(),
                ComputeError: "the GeoParquet column `{name}` has a coordinate epoch, \
                but dynamic CRSs are not supported"
            );
            if let Some(covering) = &column.covering {
                covering.bbox.column().map_err(|e| {
                    e.wrap_msg(|msg| format!("the covering of GeoParquet column `{name}`: {msg}"))
                })?;
            }
        }
        Ok(metadata)
    }

    /// The value for a file's `geo` key.
    pub fn to_json(&self) -> String {
        // Every key is a string and every number finite, so this cannot fail.
        serde_json::to_string(self).unwrap()
    }

    /// The dtype of every geometry column, and of every covering column,
    /// given the `schema` Polars reads the file with.
    pub fn dtypes(&self, schema: &Schema) -> PolarsResult<Vec<Field>> {
        let mut fields = Vec::with_capacity(self.columns.len());
        for (name, column) in &self.columns {
            let Some(dtype) = schema.get(name) else {
                polars_bail!(
                    ColumnNotFound: "the GeoParquet geometry column `{name}` is not in the file"
                );
            };
            fields.push(Field::new(name.into(), column.dtype(storage_of(dtype))?));

            let Some(covering) = &column.covering else {
                continue;
            };
            let bbox = covering.bbox.column()?;
            // The spec requires the column, but it describes nothing we read otherwise.
            if let Some(dtype) = schema.get(bbox) {
                fields.push(Field::new(bbox.into(), covering.dtype(column, dtype)?));
            }
        }
        Ok(fields)
    }
}

impl GeoParquetColumn {
    /// How a column of `dtype` is written.
    pub fn from_dtype(dtype: &DataType) -> PolarsResult<Self> {
        let (encoding, geometry_types, metadata) = match dtype {
            DataType::Extension(typ, _) if typ.name() == Encoding::Wkb.name() => (
                ColumnEncoding::Wkb,
                // Knowing them would mean reading every value.
                vec![],
                describe_encoded(dtype, Encoding::Wkb)?,
            ),
            DataType::Extension(typ, _) if typ.name() == Encoding::Wkt.name() => polars_bail!(
                SchemaMismatch: "GeoParquet cannot store WKT; \
                convert it with `from_wkt` or `to_wkb` first"
            ),
            _ => {
                let column = describe(dtype)?;
                polars_ensure!(
                    !column.dim.has_m(),
                    SchemaMismatch: "GeoParquet 1.1 cannot store M values, got: {dtype}"
                );
                let z = if column.dim.has_z() { " Z" } else { "" };
                (
                    ColumnEncoding::Native(column.kind),
                    vec![format!("{}{z}", column.kind.type_name())],
                    column.metadata.clone(),
                )
            }
        };

        let crs = match metadata.projjson()? {
            // Leaving the key out would claim the data is in OGC:CRS84.
            None => Some(None),
            Some(projjson) if is_crs84(&projjson) => None,
            Some(projjson) => Some(Some(projjson)),
        };
        let edges = match metadata.edges()? {
            None => None,
            Some(Edges::Spherical) => Some(ParquetEdges::Spherical),
            Some(edges) => polars_bail!(
                ComputeError: "GeoParquet 1.1 only has planar and spherical edges, got: {}",
                serde_json::to_value(edges).unwrap()
            ),
        };
        Ok(Self {
            encoding,
            geometry_types,
            crs,
            edges,
            epoch: None,
            covering: None,
        })
    }

    /// The metadata this column's dtype carries.
    fn metadata(&self) -> Arc<ExtensionMetadata> {
        let mut metadata = ExtensionMetadata::default();
        match &self.crs {
            None => metadata = metadata.with_crs(DEFAULT_CRS),
            Some(None) => {}
            Some(Some(projjson)) => {
                metadata = metadata.with_crs(&serde_json::to_string(projjson).unwrap())
            }
        }
        if self.edges == Some(ParquetEdges::Spherical) {
            metadata = metadata.with_edges(Edges::Spherical);
        }
        Arc::new(metadata)
    }

    /// The dtype of this column, given the `storage` Polars reads it from a file as.
    pub fn dtype(&self, storage: &DataType) -> PolarsResult<DataType> {
        let metadata = self.metadata();
        // TODO: do we need to store the dimension here? we only want to store geoarrow.point
        // anyway, not pointxy or smth

        match self.encoding {
            ColumnEncoding::Native(kind) => {
                // Not from `geometry_types`, which is allowed to be empty.
                let Some(dim) = coord::dimension_of_storage(storage, kind.nesting()) else {
                    polars_bail!(
                        SchemaMismatch: "a GeoParquet `{}` column must be stored as separated \
                        x/y[/z] coordinates, got: {storage}", kind.display()
                    );
                };
                Ok(Geo::new(kind, dim, metadata).dtype())
            }
            ColumnEncoding::Wkb => {
                polars_ensure!(
                    *storage == DataType::Binary,
                    SchemaMismatch: "a GeoParquet WKB column must be stored as binary, got: {storage}"
                );
                Ok(Encoded::new(Encoding::Wkb, metadata).dtype())
            }
        }
    }
}

impl Covering {
    fn for_column(name: &str, dtype: &DataType, bbox: &str, schema: &Schema) -> PolarsResult<Self> {
        let geometry = describe(dtype)?;
        let Some(DataType::Extension(typ, storage)) = schema.get(bbox) else {
            polars_bail!(
                SchemaMismatch: "the covering of `{name}` has to be a `{}` column `{bbox}`",
                bbox::NAME
            );
        };
        polars_ensure!(
            typ.name() == bbox::NAME && bbox::dimension_of(storage) == Some(geometry.dim),
            SchemaMismatch: "the covering of `{name}` has to be a `{}` column of the same dimension, \
            got `{bbox}`: {}", bbox::NAME, schema.get(bbox).unwrap()
        );
        let path = |field: &str| [bbox.to_owned(), field.to_owned()];
        let z = geometry.dim.has_z();
        Ok(Self {
            bbox: BboxCovering {
                xmin: path("xmin"),
                ymin: path("ymin"),
                zmin: z.then(|| path("zmin")),
                xmax: path("xmax"),
                ymax: path("ymax"),
                zmax: z.then(|| path("zmax")),
            },
        })
    }

    /// The dtype of the covering column of `geometry`, read from a file as `dtype`.
    fn dtype(&self, geometry: &GeoParquetColumn, dtype: &DataType) -> PolarsResult<DataType> {
        let column = self.bbox.column()?;
        let storage = storage_of(dtype);
        let dim = if self.bbox.zmin.is_some() {
            GeoDimension::XYZ
        } else {
            GeoDimension::XY
        };
        // By name: files in the wild (the spec's own example among them)
        // don't always keep the order the spec asks for.
        let has_bounds = match storage {
            DataType::Struct(fields) => {
                let names = dim.box_field_names();
                fields.len() == names.len()
                    && fields.iter().all(|f| {
                        f.dtype() == &DataType::Float64 && names.contains(&f.name().as_str())
                    })
            }
            _ => false,
        };
        polars_ensure!(
            has_bounds,
            SchemaMismatch: "a GeoParquet bbox covering column must be stored as a struct of \
            f64 {}, got `{column}`: {storage}", dim.box_field_names().join("/")
        );
        Ok(GeoBox::new(dim, geometry.metadata()).dtype())
    }
}

impl BboxCovering {
    fn column(&self) -> PolarsResult<&str> {
        polars_ensure!(
            self.zmin.is_some() == self.zmax.is_some(),
            ComputeError: "a bbox covering needs both `zmin` and `zmax`, or neither"
        );
        let bounds = [
            ("xmin", Some(&self.xmin)),
            ("ymin", Some(&self.ymin)),
            ("zmin", self.zmin.as_ref()),
            ("xmax", Some(&self.xmax)),
            ("ymax", Some(&self.ymax)),
            ("zmax", self.zmax.as_ref()),
        ];
        let column = &self.xmin[0];
        for (bound, path) in bounds {
            let Some([in_column, field]) = path else {
                continue;
            };
            polars_ensure!(
                in_column == column,
                ComputeError: "every bound of a bbox covering has to be in the same column, \
                got `{column}` and `{in_column}`"
            );
            polars_ensure!(
                field == bound,
                ComputeError: "the `{bound}` of a bbox covering has to be its field `{bound}`, \
                got `{field}`"
            );
        }
        Ok(column)
    }
}

impl ColumnEncoding {
    /// How the encoding is spelled in the metadata.
    fn name(self) -> &'static str {
        match self {
            ColumnEncoding::Wkb => "WKB",
            // This happens to coincide with our display names.
            // If we ever change the display names, we'll have to change this.
            ColumnEncoding::Native(kind) => kind.display(),
        }
    }
}

impl Serialize for ColumnEncoding {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for ColumnEncoding {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        if name == ColumnEncoding::Wkb.name() {
            return Ok(ColumnEncoding::Wkb);
        }
        Kind::from_display(&name)
            .map(ColumnEncoding::Native)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown encoding {name:?}")))
    }
}

/// Tells an explicit `null` apart from a missing key
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

fn storage_of(dtype: &DataType) -> &DataType {
    match dtype {
        DataType::Extension(_, storage) => storage,
        storage => storage,
    }
}

/// is does GeoParquet have to describe this?
fn is_geometry(dtype: &DataType) -> bool {
    let DataType::Extension(typ, _) = dtype else {
        return false;
    };
    let name = typ.name();
    Kind::from_name(&name).is_some() || Encoding::ALL.iter().any(|e| e.name() == name)
}

/// Whether a PROJJSON CRS is OGC:CRS84, by its `id` as the spec suggests.
fn is_crs84(projjson: &Map<String, Value>) -> bool {
    projjson
        .get("id")
        .is_some_and(|id| id["authority"] == "OGC" && id["code"] == "CRS84")
}
