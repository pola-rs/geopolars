//! The serde mappings of the arguments that `docs/Common arguments.md` documents.

use geopolars_compute::{EarlyCoordCheck, OnRowError};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(remote = "OnRowError", rename_all = "snake_case")]
pub(crate) enum OnRowErrorDef {
    Null,
    Raise,
}

#[derive(Deserialize)]
#[serde(remote = "EarlyCoordCheck", rename_all = "snake_case")]
#[expect(dead_code, reason = "no expression takes `early_coord_check` yet")]
pub(crate) enum EarlyCoordCheckDef {
    Off,
    Finite,
    FiniteAndEmpty,
}
