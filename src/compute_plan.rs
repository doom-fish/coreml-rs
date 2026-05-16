//! `MLComputePlan` summaries.

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use serde::{Deserialize, Serialize};

use crate::configuration::ModelConfiguration;
use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::ffi;

/// Broad model-type classification surfaced by `MLComputePlan`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputePlanModelType {
    /// The model structure could not be classified.
    #[default]
    Unknown,
    /// A NeuralNetwork model.
    NeuralNetwork,
    /// An ML Program model.
    Program,
    /// A Pipeline model.
    Pipeline,
}

/// Snapshot of an `MLComputePlan`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputePlan {
    /// High-level model type.
    pub model_type: ComputePlanModelType,
    /// Function names published by an ML Program model.
    #[serde(default)]
    pub function_names: Vec<String>,
    /// Number of program operations discovered while walking the structure.
    #[serde(default)]
    pub operation_count: usize,
    /// Number of neural-network layers.
    #[serde(default)]
    pub layer_count: usize,
    /// Number of pipeline sub-models, when applicable.
    #[serde(default)]
    pub pipeline_model_count: usize,
}

impl ComputePlan {
    /// Load a compute-plan summary from a compiled `.mlmodelc` bundle.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded or the compute plan is unavailable.
    pub fn load_from_url(
        path: impl AsRef<Path>,
        configuration: &ModelConfiguration,
    ) -> Result<Self, CoreMLError> {
        let path = path_to_c_string(path)?;
        let configuration_json = configuration.as_json_c_string()?;
        let mut error = ptr::null_mut();
        let mut summary = ptr::null_mut();
        let status = unsafe {
            ffi::cm_compute_plan_load_summary(
                path.as_ptr(),
                configuration_json.as_ptr(),
                &mut summary,
                &mut error,
            )
        };
        if status != ffi::status::OK || summary.is_null() {
            return Err(from_swift(status, error));
        }
        serde_json::from_str(&take_owned_c_string(summary)).map_err(|decode_error| {
            CoreMLError::ComputePlanFailed(format!(
                "failed to decode compute-plan JSON: {decode_error}"
            ))
        })
    }
}

fn path_to_c_string(path: impl AsRef<Path>) -> Result<CString, CoreMLError> {
    CString::new(path.as_ref().to_string_lossy().into_owned()).map_err(|error| {
        CoreMLError::InvalidArgument(format!("path contains an interior NUL byte: {error}"))
    })
}
