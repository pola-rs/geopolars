use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod example;
mod expr;
mod geoarrow;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// The plugin's Python module.
/// Unlike custom expressions, extension types must be explicitly
/// registered with Polars.
/// `PyInit_geopolars` is that hook.
#[pymodule]
fn geopolars(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    geoarrow::register().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to register geoarrow extension types: {e}"
        ))
    })
}
