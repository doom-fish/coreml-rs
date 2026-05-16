//! `MLModelStructure` snapshots.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

use serde::{Deserialize, Serialize};

use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::ffi;

/// High-level CoreML model-structure variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStructureKind {
    /// Neural-network graph structure.
    NeuralNetwork,
    /// ML Program structure.
    Program,
    /// Pipeline structure.
    Pipeline,
    /// A future CoreML model type not currently decoded by the crate.
    Unsupported,
}

/// Snapshot of a loaded `MLModelStructure`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelStructure {
    /// High-level structure kind.
    pub kind: ModelStructureKind,
    /// Neural-network structure, when applicable.
    #[serde(default)]
    pub neural_network: Option<ModelStructureNeuralNetwork>,
    /// Program structure, when applicable.
    #[serde(default)]
    pub program: Option<ModelStructureProgram>,
    /// Pipeline structure, when applicable.
    #[serde(default)]
    pub pipeline: Option<ModelStructurePipeline>,
}

impl ModelStructure {
    /// Decode a JSON snapshot produced by the Swift bridge.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload is not valid JSON.
    pub fn from_json_str(json: &str) -> Result<Self, CoreMLError> {
        serde_json::from_str(json).map_err(|decode_error| {
            CoreMLError::DescriptionFailed(format!(
                "failed to decode MLModelStructure JSON: {decode_error}"
            ))
        })
    }

    /// Load the structure of a compiled `.mlmodelc` bundle from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the bundle cannot be read or the runtime is too old.
    pub fn load_from_url(path: impl AsRef<Path>) -> Result<Self, CoreMLError> {
        let path = path_to_c_string(path)?;
        let mut error = ptr::null_mut();
        let mut json = ptr::null_mut();
        let status =
            unsafe { ffi::cm_model_structure_load_json(path.as_ptr(), &mut json, &mut error) };
        if status != ffi::status::OK || json.is_null() {
            return Err(from_swift(status, error));
        }
        Self::from_json_str(&take_owned_c_string(json))
    }
}

/// Snapshot of a neural-network model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureNeuralNetwork {
    /// Topologically sorted layers.
    #[serde(default)]
    pub layers: Vec<ModelStructureNeuralNetworkLayer>,
}

/// Snapshot of one neural-network layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureNeuralNetworkLayer {
    /// Layer name.
    pub name: String,
    /// Layer type (for example `conv`, `pooling`, or `elementwise`).
    #[serde(rename = "type")]
    pub layer_type: String,
    /// Input edge names.
    #[serde(default)]
    pub input_names: Vec<String>,
    /// Output edge names.
    #[serde(default)]
    pub output_names: Vec<String>,
}

/// Snapshot of a pipeline model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelStructurePipeline {
    /// Names of the child models in the pipeline.
    #[serde(default)]
    pub sub_model_names: Vec<String>,
    /// Child model structures.
    #[serde(default)]
    pub sub_models: Vec<ModelStructure>,
}

/// Snapshot of an ML Program model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureProgram {
    /// Named functions published by the program.
    #[serde(default)]
    pub functions: BTreeMap<String, ModelStructureProgramFunction>,
}

/// Snapshot of one ML Program function.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureProgramFunction {
    /// Named function inputs.
    #[serde(default)]
    pub inputs: Vec<ModelStructureProgramNamedValueType>,
    /// Active block for the function.
    pub block: ModelStructureProgramBlock,
}

/// Snapshot of an ML Program block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureProgramBlock {
    /// Named block inputs.
    #[serde(default)]
    pub inputs: Vec<ModelStructureProgramNamedValueType>,
    /// Output names.
    #[serde(default)]
    pub output_names: Vec<String>,
    /// Topologically sorted operations in the block.
    #[serde(default)]
    pub operations: Vec<ModelStructureProgramOperation>,
}

/// Snapshot of one ML Program operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureProgramOperation {
    /// Operator name (for example `conv`, `softmax`, or `while_loop`).
    pub operator_name: String,
    /// Named operation arguments.
    #[serde(default)]
    pub inputs: BTreeMap<String, ModelStructureProgramArgument>,
    /// Named outputs.
    #[serde(default)]
    pub outputs: Vec<ModelStructureProgramNamedValueType>,
    /// Nested blocks for control-flow operations.
    #[serde(default)]
    pub blocks: Vec<ModelStructureProgramBlock>,
}

/// Snapshot of one ML Program named value/type pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureProgramNamedValueType {
    /// Value name.
    pub name: String,
    /// Public description of the value type.
    pub value_type: ModelStructureProgramValueType,
}

/// Snapshot of one ML Program argument.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureProgramArgument {
    /// Bindings attached to the argument.
    #[serde(default)]
    pub bindings: Vec<ModelStructureProgramBinding>,
}

/// Snapshot of one ML Program binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureProgramBinding {
    /// Bound variable name, when the argument references another value.
    #[serde(default)]
    pub name: Option<String>,
    /// Inline constant value, when the argument is constant-folded.
    #[serde(default)]
    pub value: Option<ModelStructureProgramValue>,
}

/// Stringly snapshot of an ML Program value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureProgramValue {
    /// Framework-provided string representation.
    pub description: String,
}

/// Stringly snapshot of an ML Program value type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStructureProgramValueType {
    /// Framework-provided string representation.
    pub description: String,
}

fn path_to_c_string(path: impl AsRef<Path>) -> Result<CString, CoreMLError> {
    CString::new(path.as_ref().to_string_lossy().into_owned()).map_err(|error| {
        CoreMLError::InvalidArgument(format!("path contains an interior NUL byte: {error}"))
    })
}
