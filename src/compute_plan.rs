//! `MLComputePlan` summaries.

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use serde::{Deserialize, Serialize};

use crate::compute_device::ComputeDevice;
use crate::configuration::ModelConfiguration;
use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::ffi;
use crate::model_structure::{
    ModelStructure, ModelStructureNeuralNetworkLayer, ModelStructureProgramOperation,
};

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
        let json = load_compute_plan_json(path, configuration)?;
        serde_json::from_str(&json).map_err(|decode_error| {
            CoreMLError::ComputePlanFailed(format!(
                "failed to decode compute-plan JSON: {decode_error}"
            ))
        })
    }

    /// Load the detailed compute-plan inspection snapshot for a compiled model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded or the compute plan is unavailable.
    pub fn load_details_from_url(
        path: impl AsRef<Path>,
        configuration: &ModelConfiguration,
    ) -> Result<ComputePlanDetails, CoreMLError> {
        ComputePlanDetails::load_from_url(path, configuration)
    }
}

/// Detailed `MLComputePlan` inspection data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputePlanDetails {
    /// Backwards-compatible summary fields.
    #[serde(flatten)]
    pub summary: ComputePlan,
    /// Full `MLModelStructure` snapshot surfaced by the plan.
    pub model_structure: ModelStructure,
    /// Per-operation cost / device-usage snapshots for ML Program models.
    #[serde(default)]
    pub program_operation_plans: Vec<ComputePlanProgramOperationPlan>,
    /// Per-layer device-usage snapshots for neural-network models.
    #[serde(default)]
    pub neural_network_layer_plans: Vec<ComputePlanNeuralNetworkLayerPlan>,
}

impl ComputePlanDetails {
    /// Decode a JSON snapshot produced by the Swift bridge.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload is invalid.
    pub fn from_json_str(json: &str) -> Result<Self, CoreMLError> {
        serde_json::from_str(json).map_err(|decode_error| {
            CoreMLError::ComputePlanFailed(format!(
                "failed to decode detailed compute-plan JSON: {decode_error}"
            ))
        })
    }

    /// Load the detailed compute-plan snapshot from a compiled `.mlmodelc` bundle.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded or the compute plan is unavailable.
    pub fn load_from_url(
        path: impl AsRef<Path>,
        configuration: &ModelConfiguration,
    ) -> Result<Self, CoreMLError> {
        let json = load_compute_plan_json(path, configuration)?;
        Self::from_json_str(&json)
    }
}

/// Snapshot of `MLComputePlan.Cost` / `MLComputePlanCost`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComputePlanCost {
    /// Relative execution weight in the range `[0.0, 1.0]`.
    pub weight: f64,
}

/// Snapshot of `MLComputePlan.DeviceUsage` / `MLComputePlanDeviceUsage`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputePlanDeviceUsage {
    /// Devices that can execute the layer / operation.
    #[serde(default)]
    pub supported: Vec<ComputeDevice>,
    /// Framework-preferred execution device.
    pub preferred: ComputeDevice,
}

/// Detailed ML Program operation entry paired with its cost / device usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputePlanProgramOperationPlan {
    /// Stable path into the program structure.
    pub path: String,
    /// Operation snapshot.
    pub operation: ModelStructureProgramOperation,
    /// Estimated operation cost, when available.
    #[serde(default)]
    pub cost: Option<ComputePlanCost>,
    /// Device usage, when available.
    #[serde(default)]
    pub device_usage: Option<ComputePlanDeviceUsage>,
}

/// Detailed neural-network layer entry paired with device-usage data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputePlanNeuralNetworkLayerPlan {
    /// Stable path into the neural-network layer list.
    pub path: String,
    /// Layer snapshot.
    pub layer: ModelStructureNeuralNetworkLayer,
    /// Device usage, when available.
    #[serde(default)]
    pub device_usage: Option<ComputePlanDeviceUsage>,
}

fn path_to_c_string(path: impl AsRef<Path>) -> Result<CString, CoreMLError> {
    CString::new(path.as_ref().to_string_lossy().into_owned()).map_err(|error| {
        CoreMLError::InvalidArgument(format!("path contains an interior NUL byte: {error}"))
    })
}

fn load_compute_plan_json(
    path: impl AsRef<Path>,
    configuration: &ModelConfiguration,
) -> Result<String, CoreMLError> {
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
    Ok(take_owned_c_string(summary))
}
