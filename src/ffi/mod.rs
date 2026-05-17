//! Raw FFI declarations matching the Swift `@_cdecl` exports in
//! `swift-bridge/Sources/CoreMLBridge`.
//!
//! These functions are intentionally low-level and unsafe. Prefer the safe
//! wrappers in the parent modules.

pub mod batch_provider;
pub mod compute_device;
pub mod compute_plan;
pub mod core;
pub mod feature;
pub mod ml_array_batch_provider;
pub mod ml_custom_layer;
pub mod ml_custom_model;
pub mod ml_dictionary_feature_provider;
pub mod ml_sequence;
pub mod ml_state;
pub mod model;
pub mod model_compiler;
pub mod model_configuration;
pub mod model_description;
pub mod model_structure;
pub mod multi_array;
pub mod prediction;
pub mod update;

pub use batch_provider::*;
pub use compute_device::*;
pub use compute_plan::*;
pub use core::*;
pub use feature::*;
pub use ml_array_batch_provider::*;
pub use ml_custom_layer::*;
pub use ml_custom_model::*;
pub use ml_dictionary_feature_provider::*;
pub use ml_sequence::*;
pub use ml_state::*;
pub use model::*;
pub use model_compiler::*;
pub use model_configuration::*;
pub use model_description::*;
pub use model_structure::*;
pub use multi_array::*;
pub use prediction::*;
pub use update::*;

pub mod status {
    pub const OK: i32 = 0;
    pub const INVALID_ARGUMENT: i32 = -1;
    pub const MODEL_LOAD_FAILED: i32 = -2;
    pub const PREDICTION_FAILED: i32 = -3;
    pub const COMPILATION_FAILED: i32 = -4;
    pub const FEATURE_PROVIDER_FAILED: i32 = -5;
    pub const MULTI_ARRAY_FAILED: i32 = -6;
    pub const UNSUPPORTED: i32 = -7;
    pub const TIMED_OUT: i32 = -8;
    pub const MODEL_ASSET_FAILED: i32 = -9;
    pub const INDEX_OUT_OF_RANGE: i32 = -10;
    pub const DESCRIPTION_FAILED: i32 = -11;
    pub const COMPUTE_PLAN_FAILED: i32 = -12;
    pub const UPDATE_FAILED: i32 = -13;
    pub const STATE_FAILED: i32 = -14;
    pub const CUSTOM_LAYER_FAILED: i32 = -15;
    pub const CUSTOM_MODEL_FAILED: i32 = -16;
    pub const UNKNOWN: i32 = -99;
}
