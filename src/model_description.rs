//! `MLModelDescription` snapshots and related constraint types.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::CoreMLError;
use crate::feature::FeatureType;
use crate::ml_key::MLKey;
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

impl ParameterDescription {
    /// Reconstruct the underlying `MLKey` snapshot.
    #[must_use]
    pub fn ml_key(&self) -> MLKey {
        MLKey::from(self)
    }
}

/// Extended `MLModelDescription` snapshot that includes flexible image and tensor constraints.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DetailedModelDescription {
    /// Input feature descriptions.
    #[serde(default)]
    pub inputs: Vec<DetailedFeatureDescription>,
    /// Output feature descriptions.
    #[serde(default)]
    pub outputs: Vec<DetailedFeatureDescription>,
    /// Stateful feature descriptions.
    #[serde(default)]
    pub state_features: Vec<DetailedFeatureDescription>,
    /// Training input descriptions for updatable models.
    #[serde(default)]
    pub training_inputs: Vec<DetailedFeatureDescription>,
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

impl DetailedModelDescription {
    /// Decode a detailed JSON snapshot produced by the Swift bridge.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON payload is invalid.
    pub fn from_json_str(json: &str) -> Result<Self, CoreMLError> {
        serde_json::from_str(json).map_err(|error| {
            CoreMLError::DescriptionFailed(format!(
                "failed to decode detailed MLModelDescription JSON: {error}"
            ))
        })
    }
}

/// Detailed snapshot of one `MLFeatureDescription`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetailedFeatureDescription {
    /// Feature name.
    pub name: String,
    /// Public CoreML feature type.
    pub feature_type: FeatureType,
    /// Whether the feature is optional.
    #[serde(default)]
    pub optional: bool,
    /// Multi-array constraints, when applicable.
    #[serde(default)]
    pub multi_array_constraint: Option<DetailedMultiArrayConstraint>,
    /// Image constraints, when applicable.
    #[serde(default)]
    pub image_constraint: Option<DetailedImageConstraint>,
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

/// Detailed tensor constraint snapshot including flexible-shape metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetailedMultiArrayConstraint {
    /// Required or default shape.
    pub shape: Vec<usize>,
    /// Required CoreML data type.
    pub data_type: DataType,
    /// Flexible-shape metadata, when present.
    #[serde(default)]
    pub shape_constraint: Option<MultiArrayShapeConstraint>,
}

/// Detailed image constraint snapshot including flexible-size metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetailedImageConstraint {
    /// Required or default image width.
    pub pixels_wide: usize,
    /// Required or default image height.
    pub pixels_high: usize,
    /// Required `kCVPixelFormatType_*` value.
    pub pixel_format_type: u32,
    /// Flexible-size metadata, when present.
    #[serde(default)]
    pub size_constraint: Option<ImageSizeConstraint>,
}

/// Inclusive lower/upper bound for one dimension of a flexible CoreML range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionRange {
    /// Inclusive lower bound.
    pub lower: usize,
    /// Inclusive upper bound.
    pub upper: usize,
}

/// Snapshot of one `MLImageSize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSize {
    /// Width in pixels.
    pub pixels_wide: usize,
    /// Height in pixels.
    pub pixels_high: usize,
}

/// Kinds of `MLImageSizeConstraint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSizeConstraintType {
    /// Any image size is accepted.
    Unspecified,
    /// Only explicitly enumerated sizes are accepted.
    Enumerated,
    /// Width/height ranges define the allowed sizes.
    Range,
}

/// Detailed `MLImageSizeConstraint` snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSizeConstraint {
    /// Constraint flavor.
    pub constraint_type: ImageSizeConstraintType,
    /// Accepted width range.
    pub pixels_wide_range: DimensionRange,
    /// Accepted height range.
    pub pixels_high_range: DimensionRange,
    /// Explicitly enumerated sizes, when applicable.
    #[serde(default)]
    pub enumerated_image_sizes: Vec<ImageSize>,
}

/// Kinds of `MLMultiArrayShapeConstraint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiArrayShapeConstraintType {
    /// Any tensor shape is accepted.
    Unspecified,
    /// Only explicitly enumerated shapes are accepted.
    Enumerated,
    /// Each dimension is bounded by a range.
    Range,
}

/// Detailed `MLMultiArrayShapeConstraint` snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiArrayShapeConstraint {
    /// Constraint flavor.
    pub constraint_type: MultiArrayShapeConstraintType,
    /// Per-dimension ranges, when applicable.
    #[serde(default)]
    pub size_ranges: Vec<DimensionRange>,
    /// Explicitly enumerated allowed shapes, when applicable.
    #[serde(default)]
    pub enumerated_shapes: Vec<Vec<usize>>,
}
