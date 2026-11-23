//! The serde mappings of the arguments that `geopolars/docs/source/reference/common_arguments.rst` documents.

use geopolars_compute::{
    EarlyCoordCheck, EmptyPolygonExterior, NewHoles, NewRingOrientation, NullParts, OnRowError,
    ShortLines,
};
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
#[serde(remote = "NullParts", rename_all = "snake_case")]
pub(crate) enum NullPartsDef {
    Drop,
    Null,
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

#[derive(Deserialize)]
#[serde(remote = "NewRingOrientation", rename_all = "snake_case")]
pub(crate) enum NewRingOrientationDef {
    Keep,
    Ccw,
    Cw,
}

#[derive(Deserialize)]
#[serde(remote = "NewHoles", rename_all = "snake_case")]
pub(crate) enum NewHolesDef {
    Opposite,
    Same,
}

/// No kernel takes this: the plugin resolves it into the output kind at plan time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromoteToMulti {
    Never,
    Auto,
    Always,
}

impl PromoteToMulti {
    pub(crate) fn output_is_multi(self, input_is_multi: bool, can_split: bool) -> bool {
        input_is_multi
            || match self {
                PromoteToMulti::Never => false,
                PromoteToMulti::Auto => can_split,
                PromoteToMulti::Always => true,
            }
    }
}
