//! Async API for `coreml`.
//!
//! Enable the `async` feature to use [`Model::load_async`],
//! [`ModelCompiler::compile_async`], [`Model::predict_async`], and
//! [`Model::predict_with_state_async`].
//!
//! ```toml
//! coreml = { version = "0.3", features = ["async"] }
//! ```
//!
//! The async wrappers are executor-agnostic and can be awaited on any runtime.
//!
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # pollster::block_on(async {
//! use coreml::async_api::Model;
//! use coreml::ModelConfiguration;
//! use std::path::Path;
//!
//! let configuration = ModelConfiguration::new();
//! let _model = Model::load_async(Path::new("MyModel.mlmodelc"), Some(&configuration)).await?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # })?;
//! # Ok(())
//! # }
//! ```

pub use crate::configuration::ModelConfiguration;
pub use crate::error::CoreMLError;
pub use crate::feature_provider::FeatureProvider;
pub use crate::ml_state::MLState;
pub use crate::model::Model;
pub use crate::model_compiler::ModelCompiler;
pub use crate::prediction::PredictionOptions;
