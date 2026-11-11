use std::collections::HashMap;

use polars::prelude::{DataFrame, IntoColumn, PolarsError, Series};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::{PolarsAllocator, PyDataFrame};

mod cache;
mod common_arguments;
mod expr;
mod geoarrow;
mod geoparquet;
mod proj;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

// VERY IMPORTANT!!! NONE of these may contain polars datatypes in the type signatures
// because we **don't** bind to a specific polars version

/// See `GeoArrowType.__init__`.
#[pyfunction]
#[pyo3(signature = (*, crs = None))]
fn extension_metadata(crs: Option<String>) -> PyResult<Option<String>> {
    geoarrow::crs::MetadataKwargs { crs }
        .apply(Default::default(), "type")
        .map(|metadata| metadata.serialize())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// See `GeoArrowType._declares_crs`.
#[pyfunction]
#[pyo3(signature = (metadata))]
fn declares_crs(metadata: Option<&str>) -> PyResult<bool> {
    geoarrow::crs::ExtensionMetadata::parse(metadata)
        .map(|metadata| metadata.declares_crs())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// See `GeoArrowType._string_repr`.
/// Glue for using [geoarrow::crs::ExtensionMetadata::crs_label]
#[pyfunction]
#[pyo3(signature = (metadata))]
fn crs_label(metadata: Option<&str>) -> PyResult<Option<String>> {
    geoarrow::crs::ExtensionMetadata::parse(metadata)
        .map(|metadata| metadata.crs_label().map(str::to_owned))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// See `GeoArrowType._crs`.
#[pyfunction]
#[pyo3(signature = (metadata))]
fn crs(metadata: Option<&str>) -> PyResult<Option<String>> {
    let metadata = geoarrow::crs::ExtensionMetadata::parse(metadata)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if !metadata.declares_crs() {
        return Ok(None);
    }
    metadata
        .crs()
        .map(|crs| Some(crs.to_owned()))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// See `GeoArrowType._with_crs`.
#[pyfunction]
#[pyo3(signature = (metadata, crs))]
fn with_crs(metadata: Option<&str>, crs: &str) -> PyResult<Option<String>> {
    geoarrow::crs::ExtensionMetadata::parse(metadata)
        .map(|metadata| metadata.with_crs(crs).serialize())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// See `BoxType._longitude_turn`.
#[pyfunction]
#[pyo3(signature = (metadata))]
fn longitude_turn(py: Python<'_>, metadata: Option<&str>) -> PyResult<Option<f64>> {
    let metadata = geoarrow::crs::ExtensionMetadata::parse(metadata)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if !metadata.declares_crs() {
        return Ok(None);
    }
    metadata
        .crs()
        .and_then(proj::longitude_turn)
        .map_err(|e| polars_exception(py, e))
}

/// See `GeoArrowType._crs_bounds`.
#[pyfunction]
#[pyo3(signature = (metadata, *, allow_wrapped_longitude, within_area_of_use))]
fn crs_bounds(
    py: Python<'_>,
    metadata: Option<&str>,
    allow_wrapped_longitude: bool,
    within_area_of_use: bool,
) -> PyResult<Option<[f64; 4]>> {
    let metadata = geoarrow::crs::ExtensionMetadata::parse(metadata)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Only an area of use has to be looked up; without one, nothing is lost by skipping.
    if !metadata.declares_crs() || (metadata.is_opaque_srid() && !within_area_of_use) {
        return Ok(None);
    }
    metadata
        .crs()
        .and_then(|crs| proj::crs_bounds(crs, allow_wrapped_longitude, within_area_of_use))
        .map_err(|e| polars_exception(py, e))
}

/// See `GeoArrowType._is_equal_area`.
#[pyfunction]
#[pyo3(signature = (metadata))]
fn is_equal_area(py: Python<'_>, metadata: Option<&str>) -> PyResult<bool> {
    let metadata = geoarrow::crs::ExtensionMetadata::parse(metadata)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if !metadata.declares_crs() {
        return Ok(false);
    }
    metadata
        .crs()
        .and_then(proj::is_equal_area)
        .map_err(|e| polars_exception(py, e))
}

/// See `geopolars.io.parquet._geometry_dtypes`.
///
/// Dtypes cross as an empty frame:
/// pyo3-polars cannot convert an extension dtype on its own, but a series carries it.
#[pyfunction]
#[pyo3(signature = (geo, schema))]
fn geoparquet_dtypes(py: Python<'_>, geo: &str, schema: PyDataFrame) -> PyResult<PyDataFrame> {
    let fields = geoparquet::GeoParquetMetadata::parse(geo)
        .and_then(|metadata| metadata.dtypes(schema.0.schema()))
        .map_err(|e| polars_exception(py, e))?;
    let columns = fields
        .iter()
        .map(|field| Series::new_empty(field.name().clone(), field.dtype()).into_column())
        .collect();
    DataFrame::new(0, columns)
        .map(PyDataFrame)
        .map_err(|e| polars_exception(py, e))
}

/// See `geopolars.io.parquet._with_geo_metadata`.
/// `None` if the schema has no geometry columns.
#[pyfunction]
#[pyo3(signature = (schema, coverings = HashMap::new()))]
fn geoparquet_metadata(
    py: Python<'_>,
    schema: PyDataFrame,
    coverings: HashMap<String, String>,
) -> PyResult<Option<String>> {
    geoparquet::GeoParquetMetadata::from_schema(schema.0.schema(), &coverings)
        .map(|metadata| metadata.map(|metadata| metadata.to_json()))
        .map_err(|e| polars_exception(py, e))
}

/// `err` as the matching exception of the host `polars` package,
/// so that it can be caught as, say, `pl.exceptions.ComputeError`.
fn polars_exception(py: Python<'_>, err: PolarsError) -> PyErr {
    let name = match &err {
        PolarsError::ColumnNotFound(_) => "ColumnNotFoundError",
        PolarsError::ComputeError(_) => "ComputeError",
        PolarsError::SchemaMismatch(_) => "SchemaError",
        _ => "PolarsError",
    };
    py.import("polars.exceptions")
        .and_then(|exceptions| exceptions.getattr(name))
        .and_then(|exception| exception.call1((err.to_string(),)))
        .map_or_else(|e| e, PyErr::from_value)
}

/// The plugin's Python module.
/// Here so that extension types are explicitly registered with Polars with `PyInit_geopolars`.
#[pymodule]
fn geopolars(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Expressions don't have to be registered,
    // that hook is already by calling the plugin functions.
    m.add_function(wrap_pyfunction!(extension_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(declares_crs, m)?)?;
    m.add_function(wrap_pyfunction!(crs_label, m)?)?;
    m.add_function(wrap_pyfunction!(crs, m)?)?;
    m.add_function(wrap_pyfunction!(with_crs, m)?)?;
    m.add_function(wrap_pyfunction!(longitude_turn, m)?)?;
    m.add_function(wrap_pyfunction!(is_equal_area, m)?)?;
    m.add_function(wrap_pyfunction!(crs_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(geoparquet_dtypes, m)?)?;
    m.add_function(wrap_pyfunction!(geoparquet_metadata, m)?)?;
    geoarrow::register().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to register geoarrow extension types: {e}"
        ))
    })
}
