//! The serde mappings of the arguments that `docs/Common arguments.md` documents.

use geopolars_compute::OnRowError;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(remote = "OnRowError", rename_all = "snake_case")]
pub(crate) enum OnRowErrorDef {
    Null,
    Raise,
}
