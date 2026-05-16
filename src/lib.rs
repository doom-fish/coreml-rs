#![doc = include_str!("../README.md")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::new_without_default,
    clippy::or_fun_call,
    clippy::too_long_first_doc_paragraph,
    clippy::trivially_copy_pass_by_ref,
    clippy::unsafe_derive_deserialize,
    clippy::use_self
)]

pub mod batch_provider;
pub mod compute_plan;
pub mod configuration;
pub mod error;
pub mod feature;
pub mod feature_provider;
pub mod ffi;
pub mod ml_array_batch_provider;
pub mod ml_dictionary_feature_provider;
pub mod ml_state;
pub mod model;
pub mod model_compiler;
pub mod model_configuration;
pub mod model_description;
pub mod multi_array;
pub mod prediction;
pub mod update;

pub use compute_plan::{ComputePlan, ComputePlanModelType};
pub use configuration::{
    ComputeUnits, ModelConfiguration, ModelConfigurationBridgeSnapshot, OptimizationHints,
    ParameterValue, ReshapeFrequencyHint, SpecializationStrategy,
};
pub use error::CoreMLError;
pub use feature::{Feature, FeatureType};
pub use feature_provider::{BatchProvider, FeatureProvider};
pub use ml_array_batch_provider::MLArrayBatchProvider;
pub use ml_dictionary_feature_provider::MLDictionaryFeatureProvider;
pub use ml_state::MLState;
pub use model::Model;
pub use model_compiler::ModelCompiler;
pub use model_description::{
    DictionaryConstraint, FeatureDescription, ImageConstraint, ModelDescription,
    MultiArrayConstraint, NumericConstraint, ParameterDescription, SequenceConstraint,
    StateConstraint,
};
pub use multi_array::{DataType, MultiArray};
pub use prediction::{PredictionOptions, PredictionOptionsBridgeSnapshot};
pub use update::{
    Update, UpdateContext, UpdateEvent, UpdateProgressHandlers, UpdateResult, UpdateTaskState,
};

/// Common imports for users of this crate.
pub mod prelude {
    pub use crate::compute_plan::{ComputePlan, ComputePlanModelType};
    pub use crate::configuration::{
        ComputeUnits, ModelConfiguration, ModelConfigurationBridgeSnapshot, OptimizationHints,
        ParameterValue, ReshapeFrequencyHint, SpecializationStrategy,
    };
    pub use crate::error::CoreMLError;
    pub use crate::feature::{Feature, FeatureType};
    pub use crate::feature_provider::{BatchProvider, FeatureProvider};
    pub use crate::ml_array_batch_provider::MLArrayBatchProvider;
    pub use crate::ml_dictionary_feature_provider::MLDictionaryFeatureProvider;
    pub use crate::ml_state::MLState;
    pub use crate::model::Model;
    pub use crate::model_compiler::ModelCompiler;
    pub use crate::model_description::{
        DictionaryConstraint, FeatureDescription, ImageConstraint, ModelDescription,
        MultiArrayConstraint, NumericConstraint, ParameterDescription, SequenceConstraint,
        StateConstraint,
    };
    pub use crate::multi_array::{DataType, MultiArray};
    pub use crate::prediction::{PredictionOptions, PredictionOptionsBridgeSnapshot};
    pub use crate::update::{
        Update, UpdateContext, UpdateEvent, UpdateProgressHandlers, UpdateResult, UpdateTaskState,
    };
}
