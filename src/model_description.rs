//! `MLModelDescription` snapshots and related constraint types.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::CoreMLError;
use crate::feature::FeatureType;
use crate::multi_array::DataType;

/// Pure-Rust snapshot of `MLModelDescription`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelDescription {
    /// Input feature descriptions.
    #[serde(default)]
    pub inputs: Vec<FeatureDescription>,
    /// Output feature descriptions.
    #[serde(default)]
    pub outputs: Vec<FeatureDescription>,
    /// Stateful feature descriptions.
    #[serde(default)]
    pub state_features: Vec<FeatureDescription>,
    /// Training input descriptions for updatable models.
    #[serde(default)]
    pub training_inputs: Vec<FeatureDescription>,
    /// Model parameters and their constraints.
    #[serde(default)]
    pub parameter_descriptions: Vec<ParameterDescription>,
    /// Model metadata dictionary.
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    /// Name of the primary predicted output feature.
    #[serde(default)]
    pub predicted_feature_name: Option<String>,
    /// Name of the predicted-probabilities output feature.
    #[serde(default)]
    pub predicted_probabilities_name: Option<String>,
    /// Class labels published by the model, when present.
    #[serde(default)]
    pub class_labels: Vec<Value>,
    /// Whether the model was authored as updatable.
    #[serde(default)]
    pub is_updatable: bool,
}

impl ModelDescription {
    /// Decode a JSON snapshot produced by the Swift bridge.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON payload is invalid.
    pub fn from_json_str(json: &str) -> Result<Self, CoreMLError> {
        serde_json::from_str(json).map_err(|error| {
            CoreMLError::DescriptionFailed(format!(
                "failed to decode MLModelDescription JSON: {error}"
            ))
        })
    }
}

/// Snapshot of one `MLFeatureDescription`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureDescription {
    /// Feature name.
    pub name: String,
    /// Public CoreML feature type.
    pub feature_type: FeatureType,
    /// Whether the feature is optional.
    #[serde(default)]
    pub optional: bool,
    /// Multi-array constraints, when applicable.
    #[serde(default)]
    pub multi_array_constraint: Option<MultiArrayConstraint>,
    /// Image constraints, when applicable.
    #[serde(default)]
    pub image_constraint: Option<ImageConstraint>,
    /// Dictionary constraints, when applicable.
    #[serde(default)]
    pub dictionary_constraint: Option<DictionaryConstraint>,
    /// Sequence constraints, when applicable.
    #[serde(default)]
    pub sequence_constraint: Option<SequenceConstraint>,
    /// Stateful constraints, when applicable.
    #[serde(default)]
    pub state_constraint: Option<StateConstraint>,
}

/// Shape + element-type requirement for a multi-array feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiArrayConstraint {
    /// Required or default shape.
    pub shape: Vec<usize>,
    /// Required CoreML data type.
    pub data_type: DataType,
}

/// Size + pixel-format requirement for an image feature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageConstraint {
    /// Required or default image width.
    pub pixels_wide: usize,
    /// Required or default image height.
    pub pixels_high: usize,
    /// Required `kCVPixelFormatType_*` value.
    pub pixel_format_type: u32,
}

/// Key-type requirement for dictionary features.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DictionaryConstraint {
    /// Dictionary key type.
    pub key_type: FeatureType,
}

/// Sequence element-type and count-range requirement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceConstraint {
    /// Feature type of each sequence element.
    pub value_type: FeatureType,
    /// Lower bound of the sequence count range.
    pub count_range_lower: usize,
    /// Upper bound of the sequence count range.
    pub count_range_upper: usize,
}

/// Stateful-buffer shape + scalar type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateConstraint {
    /// Buffer shape.
    pub buffer_shape: Vec<usize>,
    /// Scalar data type.
    pub data_type: DataType,
}

/// Numeric constraints published for a model parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericConstraint {
    /// Minimum allowed value.
    #[serde(default)]
    pub min: Option<f64>,
    /// Maximum allowed value.
    #[serde(default)]
    pub max: Option<f64>,
    /// Explicitly enumerated allowed values.
    #[serde(default)]
    pub enumerated: Vec<f64>,
}

/// Snapshot of one `MLParameterDescription`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterDescription {
    /// Parameter key name.
    pub key: String,
    /// Optional key scope.
    #[serde(default)]
    pub scope: Option<String>,
    /// Default value.
    #[serde(default)]
    pub default_value: Value,
    /// Numeric constraints, when applicable.
    #[serde(default)]
    pub numeric_constraint: Option<NumericConstraint>,
}
