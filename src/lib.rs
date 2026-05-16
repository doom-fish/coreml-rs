#![doc = include_str!("../README.md")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::new_without_default,
    clippy::too_long_first_doc_paragraph,
)]

pub mod configuration;
pub mod error;
pub mod feature_provider;
pub mod ffi;
pub mod model;
pub mod multi_array;

pub use configuration::{ComputeUnits, ModelConfiguration};
pub use error::CoreMLError;
pub use feature_provider::{BatchProvider, FeatureProvider, FeatureType};
pub use model::{FeatureDescription, ImageConstraint, Model, ModelDescription, MultiArrayConstraint};
pub use multi_array::{DataType, MultiArray};

/// Common imports for users of this crate.
pub mod prelude {
    pub use crate::configuration::{ComputeUnits, ModelConfiguration};
    pub use crate::error::CoreMLError;
    pub use crate::feature_provider::{BatchProvider, FeatureProvider, FeatureType};
    pub use crate::model::{
        FeatureDescription, ImageConstraint, Model, ModelDescription, MultiArrayConstraint,
    };
    pub use crate::multi_array::{DataType, MultiArray};
}
