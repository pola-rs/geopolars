//! The serde mappings of the arguments that `docs/Common arguments.md` documents.

use geopolars_compute::OnRowError;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(remote = "OnRowError", rename_all = "snake_case")]
#[expect(dead_code, reason = "no expression takes `on_row_error` yet")]
pub(crate) enum OnRowErrorDef {
    Null,
    Drop,
    Raise,
}
