//! `MLModelConfiguration` builder types.

use std::collections::BTreeMap;
use std::ffi::CString;

use serde::{Deserialize, Serialize};

use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::ffi;

/// CoreML compute-unit selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeUnits {
    /// Force inference onto CPU only.
    CpuOnly,
    /// Allow CPU and GPU execution.
    CpuAndGpu,
    /// Allow CPU and Apple Neural Engine execution.
    CpuAndNeuralEngine,
    /// Let CoreML use the best available compute units.
    #[default]
    All,
}

/// Hint for how often flexible-shape models will change shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReshapeFrequencyHint {
    /// Optimize for fast switching between shapes.
    #[default]
    Frequent,
    /// Optimize for repeated predictions at the same shape.
    Infrequent,
}

/// Specialization strategy for CoreML model loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecializationStrategy {
    /// CoreML's default strategy.
    #[default]
    Default,
    /// Prefer faster steady-state prediction latency.
    FastPrediction,
}

/// Optional optimization hints for `MLModelConfiguration`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationHints {
    /// Expected reshape frequency.
    pub reshape_frequency: ReshapeFrequencyHint,
    /// Optional specialization strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub specialization_strategy: Option<SpecializationStrategy>,
}

impl Default for OptimizationHints {
    fn default() -> Self {
        Self {
            reshape_frequency: ReshapeFrequencyHint::Frequent,
            specialization_strategy: None,
        }
    }
}

/// JSON-friendly parameter value for `MLModelConfiguration.parameters`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParameterValue {
    /// Boolean parameter.
    Bool(bool),
    /// Signed 64-bit integer parameter.
    Int64(i64),
    /// Double-precision floating-point parameter.
    Double(f64),
    /// String parameter.
    String(String),
}

impl From<bool> for ParameterValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for ParameterValue {
    fn from(value: i64) -> Self {
        Self::Int64(value)
    }
}

impl From<f64> for ParameterValue {
    fn from(value: f64) -> Self {
        Self::Double(value)
    }
}

impl From<String> for ParameterValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for ParameterValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

/// Safe Rust builder for `MLModelConfiguration`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelConfiguration {
    pub(crate) compute_units: ComputeUnits,
    pub(crate) allow_low_precision_accumulation_on_gpu: bool,
    pub(crate) display_name: Option<String>,
    pub(crate) function_name: Option<String>,
    pub(crate) optimization_hints: Option<OptimizationHints>,
    #[serde(default)]
    pub(crate) parameters: BTreeMap<String, ParameterValue>,
}

/// Snapshot of the bridge-normalized configuration.
pub type ModelConfigurationBridgeSnapshot = ModelConfiguration;

impl ModelConfiguration {
    /// Create a configuration with CoreML defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the compute units CoreML may use.
    #[must_use]
    pub fn with_compute_units(mut self, units: ComputeUnits) -> Self {
        self.compute_units = units;
        self
    }

    /// Allow lower-precision accumulation on GPU when supported.
    #[must_use]
    pub fn with_allow_low_precision_accumulation_on_gpu(mut self, allow: bool) -> Self {
        self.allow_low_precision_accumulation_on_gpu = allow;
        self
    }

    /// Set the human-readable display name for the loaded model instance.
    #[must_use]
    pub fn with_display_name(mut self, name: &str) -> Self {
        self.display_name = Some(name.to_owned());
        self
    }

    /// Select a named function inside a multi-function model asset.
    #[must_use]
    pub fn with_function_name(mut self, function_name: &str) -> Self {
        self.function_name = Some(function_name.to_owned());
        self
    }

    /// Supply CoreML optimization hints.
    #[must_use]
    pub fn with_optimization_hints(mut self, hints: OptimizationHints) -> Self {
        self.optimization_hints = Some(hints);
        self
    }

    /// Set one model/update parameter by name.
    #[must_use]
    pub fn with_parameter(
        mut self,
        key: impl Into<String>,
        value: impl Into<ParameterValue>,
    ) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Current compute-unit selection.
    #[must_use]
    pub const fn compute_units(&self) -> ComputeUnits {
        self.compute_units
    }

    /// Whether low-precision GPU accumulation is enabled.
    #[must_use]
    pub const fn allow_low_precision_accumulation_on_gpu(&self) -> bool {
        self.allow_low_precision_accumulation_on_gpu
    }

    /// Optional display name supplied at load time.
    #[must_use]
    pub fn display_name(&self) -> Option<&str> {
        self.display_name.as_deref()
    }

    /// Optional function name selected for multi-function assets.
    #[must_use]
    pub fn function_name(&self) -> Option<&str> {
        self.function_name.as_deref()
    }

    /// Optional optimization hints.
    #[must_use]
    pub fn optimization_hints(&self) -> Option<&OptimizationHints> {
        self.optimization_hints.as_ref()
    }

    /// Load/update parameters keyed by Apple parameter names.
    #[must_use]
    pub fn parameters(&self) -> &BTreeMap<String, ParameterValue> {
        &self.parameters
    }

    /// Round-trip this configuration through the Swift bridge.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization or bridge decoding fails.
    pub fn bridge_snapshot(&self) -> Result<ModelConfigurationBridgeSnapshot, CoreMLError> {
        let json = self.as_json_c_string()?;
        let mut error = std::ptr::null_mut();
        let snapshot =
            unsafe { ffi::cm_model_configuration_snapshot_json(json.as_ptr(), &mut error) };
        if snapshot.is_null() {
            return Err(from_swift(ffi::status::MODEL_LOAD_FAILED, error));
        }
        serde_json::from_str(&take_owned_c_string(snapshot)).map_err(|decode_error| {
            CoreMLError::ModelLoadFailed(format!(
                "failed to decode model-configuration snapshot JSON: {decode_error}"
            ))
        })
    }

    pub(crate) fn as_json_c_string(&self) -> Result<CString, CoreMLError> {
        CString::new(serde_json::to_string(self).map_err(|error| {
            CoreMLError::InvalidArgument(format!("failed to encode model configuration: {error}"))
        })?)
        .map_err(|error| {
            CoreMLError::InvalidArgument(format!(
                "model configuration JSON contained an interior NUL byte: {error}"
            ))
        })
    }
}
