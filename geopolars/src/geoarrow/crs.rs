//! Home of [ExtensionMetadata].
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use geoarrow_schema::{Crs, CrsType, Edges};
use polars::prelude::*;
use proj::Proj;
use serde::Deserialize;
use serde_json::{Map, Value};

/// Metadata for GeoArrow columns.
///
/// GeoArrow keeps metadata (such as the CRS) in `ARROW:extension:metadata`.
/// It is encoded as a JSON string, looking something like:
/// ```json
/// {"crs": ..., "crs_type": ..., "edges": ...}
/// ```
///
/// In GeoPolars, we keep this data as-is, but wrapped in a newtype [ExtensionMetadata].
/// This allows us to reconstruct it **losslessly** for fields we do not support, but others might,
/// while also giving an anchoring point for functionality that has to do with the metadata.
/// This is a slight deviation from the GeoArrow schema,
/// but makes chaining with proprietary data producers more versatile.
///
/// whitespace and key order do not make two metadata different,
/// and [`serialize`](Self::serialize) always writes compact JSON.
///
/// The CRS and edges are parsed once, when the metadata is built, and cached next to the map.
/// The map stays the source of truth: only it is compared, hashed and serialized.
#[derive(Clone, Debug)]
pub struct ExtensionMetadata {
    map: Map<String, Value>,
    /// Cache of `map`: the CRS in a form PROJ accepts, or why there is none.
    crs: Result<String, String>,
    /// Cache of `map`: how the edges run, or why they are invalid.
    edges: Result<Option<Edges>, String>,
    /// Cache of `map`: a short name for the CRS, to display. `None` without one.
    crs_label: Option<String>,
}

impl PartialEq for ExtensionMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
    }
}

impl Eq for ExtensionMetadata {}

impl Hash for ExtensionMetadata {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.map.hash(state)
    }
}

impl Default for ExtensionMetadata {
    fn default() -> Self {
        Self::from_map(Map::new())
    }
}

impl ExtensionMetadata {
    /// The only constructor, so the caches never drift from the map.
    fn from_map(map: Map<String, Value>) -> Self {
        Self {
            crs: crs_of(&map),
            edges: edges_of(&map),
            crs_label: crs_label_of(&map),
            map,
        }
    }

    /// Parses the raw metadata string.
    /// If no metadata is found, it produces an empty object.
    pub fn parse(metadata: Option<&str>) -> PolarsResult<Self> {
        // Some producers write an empty string rather than leaving it out.
        let metadata = match metadata.map(str::trim) {
            None | Some("") => return Ok(Self::default()),
            Some(metadata) => metadata,
        };
        match serde_json::from_str(metadata) {
            Ok(Value::Object(map)) => Ok(Self::from_map(map)),
            Ok(_) => polars_bail!(
                ComputeError: "GeoArrow extension metadata must be a JSON object, got: {metadata}"
            ),
            Err(e) => polars_bail!(
                ComputeError: "invalid GeoArrow extension metadata {metadata}: {e}"
            ),
        }
    }

    /// The metadata in a JSON-formatted string.
    /// If empty, this returns `None`.
    pub fn serialize(&self) -> Option<String> {
        (!self.map.is_empty()).then(|| serde_json::to_string(&self.map).unwrap())
    }

    /// Gets the CRS in a form PROJ accepts
    ///
    /// Errors if there is no CRS.
    pub fn crs(&self) -> PolarsResult<&str> {
        self.crs
            .as_deref()
            .map_err(|e| polars_err!(ComputeError: "{e}"))
    }

    /// A short name for the CRS, to display: a PROJJSON CRS by its id or name.
    /// `None` if the metadata declares no CRS.
    pub fn crs_label(&self) -> Option<&str> {
        self.crs_label.as_deref()
    }

    /// Whether the metadata names a CRS at all (whether or not PROJ can use it).
    pub fn declares_crs(&self) -> bool {
        self.map.get("crs").is_some_and(|crs| !crs.is_null())
    }

    /// Whether the CRS is an SRID, which only its producer can read.
    pub fn is_opaque_srid(&self) -> bool {
        self.map.get("crs_type").and_then(Value::as_str) == Some("srid")
    }

    /// Consuming setter for CRS.
    pub fn with_crs(self, crs: &str) -> Self {
        let mut map = self.map;
        map.insert("crs".to_owned(), crs_value(crs));
        map.shift_remove("crs_type");
        Self::from_map(map)
    }

    /// The CRS as a PROJJSON object.
    /// This is used for GeoParquet.
    /// `None` if the metadata declares no CRS.
    /// A PROJJSON CRS comes back as-is; any other form is converted by PROJ.
    pub fn projjson(&self) -> PolarsResult<Option<Map<String, Value>>> {
        match self.map.get("crs") {
            None | Some(Value::Null) => return Ok(None),
            Some(Value::Object(projjson)) => return Ok(Some(projjson.clone())),
            Some(_) => {}
        }
        let crs = self.crs()?;
        let projjson = Proj::new(crs)
            .map_err(|e| polars_err!(ComputeError: "PROJ cannot read the CRS {crs}: {e}"))?
            .to_projjson(None, None, None)
            .map_err(
                |e| polars_err!(ComputeError: "PROJ cannot write the CRS {crs} as PROJJSON: {e}"),
            )?;
        match serde_json::from_str(&projjson) {
            Ok(Value::Object(projjson)) => Ok(Some(projjson)),
            _ => polars_bail!(
                ComputeError: "PROJ wrote invalid PROJJSON for the CRS {crs}: {projjson}"
            ),
        }
    }

    /// How the edges between coordinates run. `None` means planar.
    pub fn edges(&self) -> PolarsResult<Option<Edges>> {
        self.edges
            .clone()
            .map_err(|e| polars_err!(ComputeError: "{e}"))
    }

    /// Consuming setter for edges.
    pub fn with_edges(self, edges: Edges) -> Self {
        // A unit variant always serialises.
        let edges = serde_json::to_value(edges).unwrap();
        let mut map = self.map;
        map.insert("edges".to_owned(), edges);
        Self::from_map(map)
    }
}

/// What [`ExtensionMetadata::crs`] gives for `map`, with the error as a message.
fn crs_of(map: &Map<String, Value>) -> Result<String, String> {
    let crs = Crs::deserialize(map).map_err(|e| {
        let metadata = serde_json::to_string(map).unwrap();
        format!("invalid CRS in GeoArrow metadata {metadata}: {e}")
    })?;

    match (crs.crs_value(), crs.crs_type()) {
        (None, _) => Err("the column does not declare a CRS, so it cannot be reprojected".into()),
        // Only the producer knows what an SRID refers to.
        (Some(srid), Some(CrsType::Srid)) => Err(format!(
            "the column's CRS is the opaque SRID {srid}, which cannot be reprojected from"
        )),
        // PROJJSON, WKT, `AUTH:CODE`, or a string of unstated kind: PROJ gets to try.
        (Some(Value::String(crs)), _) => Ok(crs.clone()),
        (Some(projjson @ Value::Object(_)), _) => Ok(projjson.to_string()),
        (Some(other), _) => Err(format!(
            "a CRS must be a PROJJSON object or a string, got: {other}"
        )),
    }
}

/// What [`ExtensionMetadata::crs_label`] gives for `map`.
/// Whether PROJ can use the CRS does not matter.
fn crs_label_of(map: &Map<String, Value>) -> Option<String> {
    match map.get("crs")? {
        Value::Null => None,
        Value::String(crs) => Some(crs.clone()),
        Value::Object(projjson) => {
            let id = projjson.get("id").and_then(|id| {
                let authority = id.get("authority")?.as_str()?;
                match id.get("code")? {
                    Value::String(code) => Some(format!("{authority}:{code}")),
                    code @ Value::Number(_) => Some(format!("{authority}:{code}")),
                    _ => None,
                }
            });
            let name = || projjson.get("name")?.as_str().map(str::to_owned);
            Some(
                id.or_else(name)
                    .unwrap_or_else(|| Value::Object(projjson.clone()).to_string()),
            )
        }
        other => Some(other.to_string()),
    }
}

/// What [`ExtensionMetadata::edges`] gives for `map`, with the error as a message.
fn edges_of(map: &Map<String, Value>) -> Result<Option<Edges>, String> {
    match map.get("edges") {
        None | Some(Value::Null) => Ok(None),
        Some(edges) => Edges::deserialize(edges)
            .map(Some)
            .map_err(|e| format!("invalid edges {edges} in GeoArrow metadata: {e}")),
    }
}

fn crs_value(crs: &str) -> Value {
    match serde_json::from_str(crs) {
        Ok(object @ Value::Object(_)) => object,
        _ => Value::String(crs.to_owned()),
    }
}

/// The metadata fields a constructor for
/// one of our Arrow Extension types in the API can be asked to set, as keyword arguments.
/// These form a fixed set, not arbitrary keys.
/// These have to be mirrored on the python side.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetadataKwargs {
    /// The CRS the coordinates are in, in any form PROJ accepts.
    pub crs: Option<String>,
}

impl MetadataKwargs {
    /// The metadata of a geometry built from parts carrying `metadata`.
    /// If empty, `metadata` comes back untouched.
    ///
    /// This labels and never reprojects. `metadata` that already declares a different
    /// CRS is an error rather than being relabelled into the wrong place.
    pub fn apply(
        &self,
        metadata: Arc<ExtensionMetadata>,
        name_in_err_msg: &str,
    ) -> PolarsResult<Arc<ExtensionMetadata>> {
        let Some(crs) = &self.crs else {
            return Ok(metadata);
        };

        match metadata.map.get("crs") {
            None | Some(Value::Null) => Ok(Arc::new(Arc::unwrap_or_clone(metadata).with_crs(crs))),
            Some(declared) if *declared == crs_value(crs) => Ok(metadata),
            Some(declared) => polars_bail!(
                SchemaMismatch: "the CRS of the {name_in_err_msg} is already {declared}, not {}; \
                use `to_crs` to reproject instead",
                crs_value(crs)
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Only what the Python suite cannot reach: metadata other producers write.

    use super::*;

    fn crs_of(metadata: Option<&str>) -> PolarsResult<String> {
        ExtensionMetadata::parse(metadata)?.crs().map(str::to_owned)
    }

    fn with_crs(metadata: Option<&str>, crs: &str) -> Option<String> {
        ExtensionMetadata::parse(metadata)
            .unwrap()
            .with_crs(crs)
            .serialize()
    }

    #[test]
    fn reads_every_crs_representation() {
        let projjson = r#"{"crs":{"type":"GeographicCRS","name":"WGS 84"},"crs_type":"projjson"}"#;
        assert_eq!(
            crs_of(Some(projjson)).unwrap(),
            r#"{"type":"GeographicCRS","name":"WGS 84"}"#
        );
        let code = r#"{"crs":"EPSG:4326","crs_type":"authority_code"}"#;
        assert_eq!(crs_of(Some(code)).unwrap(), "EPSG:4326");
        assert_eq!(crs_of(Some(r#"{"crs":"OGC:CRS84"}"#)).unwrap(), "OGC:CRS84");
    }

    #[test]
    fn refuses_what_proj_cannot_use() {
        for metadata in [
            None,
            Some(""),
            Some("{}"),
            Some(r#"{"edges":"spherical"}"#),
            Some(r#"{"crs":null}"#),
            Some(r#"{"crs":"4326","crs_type":"srid"}"#),
            Some(r#"{"crs":4326}"#),
            Some(r#"["crs"]"#),
            Some("not json"),
        ] {
            assert!(crs_of(metadata).is_err(), "{metadata:?}");
        }
    }

    #[test]
    fn keeps_every_other_key_in_place() {
        let metadata = r#"{"vendor":{"a":[1,2]},"crs":"EPSG:4326","crs_type":"authority_code","edges":"karney","zz":true}"#;
        assert_eq!(
            with_crs(Some(metadata), "EPSG:3857").unwrap(),
            r#"{"vendor":{"a":[1,2]},"crs":"EPSG:3857","edges":"karney","zz":true}"#
        );
        // Even keys and values GeoArrow does not define.
        let metadata = r#"{"edges":"geodesic-ish","x-extra":1}"#;
        assert_eq!(
            with_crs(Some(metadata), "EPSG:3857").unwrap(),
            r#"{"edges":"geodesic-ish","x-extra":1,"crs":"EPSG:3857"}"#
        );
    }

    #[test]
    fn writes_a_json_crs_unescaped() {
        let out = with_crs(None, r#"{"type": "GeographicCRS"}"#).unwrap();
        assert_eq!(out, r#"{"crs":{"type":"GeographicCRS"}}"#);
        assert_eq!(crs_of(Some(&out)).unwrap(), r#"{"type":"GeographicCRS"}"#);
    }

    fn apply(metadata: Option<&str>, crs: Option<&str>) -> PolarsResult<Option<String>> {
        let kwargs = MetadataKwargs {
            crs: crs.map(str::to_owned),
        };
        let metadata = Arc::new(ExtensionMetadata::parse(metadata)?);
        Ok(kwargs.apply(metadata, "parts")?.serialize())
    }

    #[test]
    fn nothing_to_set_keeps_every_key() {
        let metadata = r#"{ "zz" : true, "crs" : "EPSG:4326", "x-extra": 1 }"#;
        assert_eq!(
            apply(Some(metadata), None).unwrap().unwrap(),
            r#"{"zz":true,"crs":"EPSG:4326","x-extra":1}"#
        );
        assert_eq!(apply(None, None).unwrap(), None);
    }

    /// This must hold, otherwise the python side might start disagreeing.
    #[test]
    fn spelling_does_not_make_metadata_different() {
        let a = ExtensionMetadata::parse(Some(r#"{"crs":"EPSG:4326","edges":"karney"}"#));
        let b = ExtensionMetadata::parse(Some(r#"{ "edges": "karney", "crs": "EPSG:4326" }"#));
        assert_eq!(a.unwrap(), b.unwrap());
    }

    #[test]
    fn sets_a_crs_next_to_the_other_keys() {
        assert_eq!(
            apply(None, Some("EPSG:4326")).unwrap().unwrap(),
            r#"{"crs":"EPSG:4326"}"#
        );
        assert_eq!(
            apply(Some(r#"{"edges":"spherical"}"#), Some("EPSG:4326"))
                .unwrap()
                .unwrap(),
            r#"{"edges":"spherical","crs":"EPSG:4326"}"#
        );
    }

    #[test]
    fn the_same_crs_again_is_not_a_conflict() {
        // Untouched: `crs_type` survives, where `with_crs` would have dropped it.
        let code = r#"{"crs":"EPSG:4326","crs_type":"authority_code"}"#;
        assert_eq!(apply(Some(code), Some("EPSG:4326")).unwrap().unwrap(), code);
        let projjson = r#"{"crs":{"type":"GeographicCRS"}}"#;
        assert_eq!(
            apply(Some(projjson), Some(r#"{ "type": "GeographicCRS" }"#))
                .unwrap()
                .unwrap(),
            projjson
        );
    }

    #[test]
    fn refuses_to_relabel_a_declared_crs() {
        let err = apply(Some(r#"{"crs":"EPSG:4326"}"#), Some("EPSG:28992")).unwrap_err();
        assert!(err.to_string().contains("to_crs"), "{err}");
    }

    #[test]
    fn no_keys_is_no_metadata() {
        assert_eq!(
            ExtensionMetadata::parse(Some("{}")).unwrap().serialize(),
            None
        );
    }
}
