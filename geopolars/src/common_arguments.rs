//! The serde mappings of the arguments that `docs/Common arguments.md` documents.

use geopolars_compute::{EarlyCoordCheck, EmptyPolygonExterior, OnRowError, ShortLines};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(remote = "OnRowError", rename_all = "snake_case")]
pub(crate) enum OnRowErrorDef {
    Null,
    Raise,
}

#[derive(Deserialize)]
#[serde(remote = "EarlyCoordCheck", rename_all = "snake_case")]
pub(crate) enum EarlyCoordCheckDef {
    Off,
    Finite,
    FiniteAndEmpty,
}

#[derive(Deserialize)]
#[serde(remote = "ShortLines", rename_all = "snake_case")]
pub(crate) enum ShortLinesDef {
    Null,
    Keep,
    Empty,
}

#[derive(Deserialize)]
#[serde(remote = "EmptyPolygonExterior", rename_all = "snake_case")]
pub(crate) enum EmptyPolygonExteriorDef {
    Keep,
    Null,
    Error,
    Empty,
}
