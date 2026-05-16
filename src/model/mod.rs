//! `MLModel` loading, compilation, description, and inference.

use core::ffi::c_void;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr;

use serde::Deserialize;
use serde_json::Value;

use crate::configuration::ModelConfiguration;
use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::feature_provider::{BatchProvider, FeatureProvider, FeatureType};
use crate::ffi;
use crate::multi_array::DataType;

/// Pure-Rust snapshot of `MLModelDescription`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelDescription {
    /// Input feature descriptions.
    #[serde(default)]
    pub inputs: Vec<FeatureDescription>,
    /// Output feature descriptions.
    #[serde(default)]
    pub outputs: Vec<FeatureDescription>,
    /// Model metadata dictionary.
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    /// Whether the model was authored as updatable.
    #[serde(default)]
    pub is_updatable: bool,
}

/// Snapshot of one `MLFeatureDescription`.
#[derive(Debug, Clone, Deserialize)]
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
}

/// Shape + element-type requirement for a multi-array feature.
#[derive(Debug, Clone, Deserialize)]
pub struct MultiArrayConstraint {
    /// Required or default shape.
    pub shape: Vec<usize>,
    /// Required CoreML data type.
    pub data_type: DataType,
}

/// Size + pixel-format requirement for an image feature.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageConstraint {
    /// Required or default image width.
    pub pixels_wide: usize,
    /// Required or default image height.
    pub pixels_high: usize,
    /// Required `kCVPixelFormatType_*` value.
    pub pixel_format_type: u32,
}

/// Owned `MLModel` handle.
pub struct Model {
    ptr: *mut c_void,
}

impl Model {
    /// Load a compiled `.mlmodelc` bundle from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is invalid or CoreML rejects the model.
    pub fn load_from_url(
        path: impl AsRef<Path>,
        configuration: &ModelConfiguration,
    ) -> Result<Self, CoreMLError> {
        let path = path_to_c_string(path)?;
        let display_name = option_to_c_string(configuration.display_name())?;
        let mut error = ptr::null_mut();
        let mut model = ptr::null_mut();
        let status = unsafe {
            ffi::cm_model_load(
                path.as_ptr(),
                configuration.compute_units.as_ffi(),
                configuration.allow_low_precision_accumulation_on_gpu,
                display_name.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                &mut model,
                &mut error,
            )
        };
        if status != ffi::status::OK || model.is_null() {
            return Err(from_swift(status, error));
        }
        Ok(Self { ptr: model })
    }

    /// Load a model directly from `.mlmodel` specification bytes using `MLModelAsset`.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are empty or CoreML rejects the specification.
    pub fn load_from_specification_data(
        specification: &[u8],
        configuration: &ModelConfiguration,
    ) -> Result<Self, CoreMLError> {
        if specification.is_empty() {
            return Err(CoreMLError::InvalidArgument(
                "model specification must not be empty".to_owned(),
            ));
        }

        let display_name = option_to_c_string(configuration.display_name())?;
        let mut error = ptr::null_mut();
        let mut model = ptr::null_mut();
        let status = unsafe {
            ffi::cm_model_load_from_specification(
                specification.as_ptr(),
                specification.len(),
                configuration.compute_units.as_ffi(),
                configuration.allow_low_precision_accumulation_on_gpu,
                display_name.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                &mut model,
                &mut error,
            )
        };
        if status != ffi::status::OK || model.is_null() {
            return Err(from_swift(status, error));
        }
        Ok(Self { ptr: model })
    }

    /// Compile a source `.mlmodel` file to a temporary `.mlmodelc` bundle.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML cannot compile the source model.
    pub fn compile_model(mlmodel_path: impl AsRef<Path>) -> Result<PathBuf, CoreMLError> {
        let path = path_to_c_string(mlmodel_path)?;
        let mut error = ptr::null_mut();
        let mut compiled_path = ptr::null_mut();
        let status = unsafe { ffi::cm_model_compile(path.as_ptr(), &mut compiled_path, &mut error) };
        if status != ffi::status::OK || compiled_path.is_null() {
            return Err(from_swift(status, error));
        }
        Ok(PathBuf::from(take_owned_c_string(compiled_path)))
    }

    /// Compile a source `.mlmodel` then load the resulting compiled bundle.
    ///
    /// # Errors
    ///
    /// Returns an error if compilation or loading fails.
    pub fn compile_and_load(
        mlmodel_path: impl AsRef<Path>,
        configuration: &ModelConfiguration,
    ) -> Result<Self, CoreMLError> {
        let compiled = Self::compile_model(mlmodel_path)?;
        Self::load_from_url(compiled, configuration)
    }

    /// Snapshot the model description.
    #[must_use]
    pub fn description(&self) -> ModelDescription {
        let json = unsafe { ffi::cm_model_description_json(self.ptr) };
        serde_json::from_str(&take_owned_c_string(json)).unwrap_or_default()
    }

    /// Run one synchronous prediction.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML rejects the input feature provider or inference fails.
    pub fn predict(&self, inputs: &FeatureProvider) -> Result<FeatureProvider, CoreMLError> {
        let mut error = ptr::null_mut();
        let mut out = ptr::null_mut();
        let status = unsafe { ffi::cm_model_predict(self.ptr, inputs.ptr, &mut out, &mut error) };
        if status != ffi::status::OK || out.is_null() {
            return Err(from_swift(status, error));
        }
        FeatureProvider::from_raw(out).ok_or_else(|| {
            CoreMLError::PredictionFailed("CoreML prediction returned no outputs".to_owned())
        })
    }

    /// Run one synchronous batch prediction.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML rejects the input batch or inference fails.
    pub fn predict_batch(&self, inputs: &BatchProvider) -> Result<BatchProvider, CoreMLError> {
        let mut error = ptr::null_mut();
        let mut out = ptr::null_mut();
        let status = unsafe { ffi::cm_model_predict_batch(self.ptr, inputs.ptr, &mut out, &mut error) };
        if status != ffi::status::OK || out.is_null() {
            return Err(from_swift(status, error));
        }
        BatchProvider::from_raw(out).ok_or_else(|| {
            CoreMLError::PredictionFailed("CoreML batch prediction returned no outputs".to_owned())
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cm_object_release(self.ptr) };
        }
    }
}

impl core::fmt::Debug for Model {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Model")
            .field("description", &self.description())
            .finish()
    }
}

fn path_to_c_string(path: impl AsRef<Path>) -> Result<CString, CoreMLError> {
    CString::new(path.as_ref().to_string_lossy().into_owned()).map_err(|error| {
        CoreMLError::InvalidArgument(format!(
            "path contains an interior NUL byte: {error}"
        ))
    })
}

fn option_to_c_string(value: Option<&str>) -> Result<Option<CString>, CoreMLError> {
    value
        .map(|value| {
            CString::new(value).map_err(|error| {
                CoreMLError::InvalidArgument(format!(
                    "display name contains an interior NUL byte: {error}"
                ))
            })
        })
        .transpose()
}
