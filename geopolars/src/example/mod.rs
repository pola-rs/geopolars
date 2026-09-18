//! The pyo3-polars example plugin this repo was forked from.
//!
//! Pig latin, jaccard, haversine, a deliberate `todo!()` panic. Kept as worked
//! examples of kwargs structs, `output_type_func` and rayon parallelism. New
//! geometry work goes in [`crate::geoarrow`] and [`crate::expr`].

pub mod distances;
pub mod expressions;
