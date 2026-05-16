//! `MLModel` loading, description, compilation, and inference.

use core::ffi::c_void;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

use crate::configuration::ModelConfiguration;
use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::feature_provider::{BatchProvider, FeatureProvider};
use crate::ffi;
use crate::ml_state::MLState;
use crate::model_compiler::ModelCompiler;
pub use crate::model_description::{
    DictionaryConstraint, FeatureDescription, ImageConstraint, ModelDescription,
    MultiArrayConstraint, NumericConstraint, ParameterDescription, SequenceConstraint,
    StateConstraint,
};
use crate::prediction::PredictionOptions;

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
        let configuration_json = configuration.as_json_c_string()?;
        let mut error = ptr::null_mut();
        let mut model = ptr::null_mut();
        let status = unsafe {
            ffi::cm_model_load(
                path.as_ptr(),
                configuration_json.as_ptr(),
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

        let configuration_json = configuration.as_json_c_string()?;
        let mut error = ptr::null_mut();
        let mut model = ptr::null_mut();
        let status = unsafe {
            ffi::cm_model_load_from_specification(
                specification.as_ptr(),
                specification.len(),
                configuration_json.as_ptr(),
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
    pub fn compile_model(
        mlmodel_path: impl AsRef<Path>,
    ) -> Result<std::path::PathBuf, CoreMLError> {
        ModelCompiler::compile(mlmodel_path)
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
        ModelCompiler::compile_and_load(mlmodel_path, configuration)
    }

    /// Snapshot the model description.
    #[must_use]
    pub fn description(&self) -> ModelDescription {
        let json = unsafe { ffi::cm_model_description_json(self.ptr) };
        if json.is_null() {
            return ModelDescription::default();
        }
        ModelDescription::from_json_str(&take_owned_c_string(json)).unwrap_or_default()
    }

    /// Run one synchronous prediction with default options.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML rejects the input feature provider or inference fails.
    pub fn predict(&self, inputs: &FeatureProvider) -> Result<FeatureProvider, CoreMLError> {
        let options = PredictionOptions::default();
        self.predict_with_options(inputs, &options)
    }

    /// Run one synchronous prediction with explicit options.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML rejects the input feature provider or inference fails.
    pub fn predict_with_options(
        &self,
        inputs: &FeatureProvider,
        options: &PredictionOptions,
    ) -> Result<FeatureProvider, CoreMLError> {
        let options_json = options.as_json_c_string()?;
        let mut error = ptr::null_mut();
        let mut out = ptr::null_mut();
        let status = unsafe {
            ffi::cm_model_predict_with_options(
                self.ptr,
                inputs.ptr,
                options_json.as_ptr(),
                &mut out,
                &mut error,
            )
        };
        if status != ffi::status::OK || out.is_null() {
            return Err(from_swift(status, error));
        }
        FeatureProvider::from_raw(out).ok_or_else(|| {
            CoreMLError::PredictionFailed("CoreML prediction returned no outputs".to_owned())
        })
    }

    /// Run one synchronous batch prediction with default options.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML rejects the input batch or inference fails.
    pub fn predict_batch(&self, inputs: &BatchProvider) -> Result<BatchProvider, CoreMLError> {
        let options = PredictionOptions::default();
        self.predict_batch_with_options(inputs, &options)
    }

    /// Run one synchronous batch prediction with explicit options.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML rejects the input batch or inference fails.
    pub fn predict_batch_with_options(
        &self,
        inputs: &BatchProvider,
        options: &PredictionOptions,
    ) -> Result<BatchProvider, CoreMLError> {
        let options_json = options.as_json_c_string()?;
        let mut error = ptr::null_mut();
        let mut out = ptr::null_mut();
        let status = unsafe {
            ffi::cm_model_predict_batch_with_options(
                self.ptr,
                inputs.ptr,
                options_json.as_ptr(),
                &mut out,
                &mut error,
            )
        };
        if status != ffi::status::OK || out.is_null() {
            return Err(from_swift(status, error));
        }
        BatchProvider::from_raw(out).ok_or_else(|| {
            CoreMLError::PredictionFailed("CoreML batch prediction returned no outputs".to_owned())
        })
    }

    /// Create a new CoreML state object for stateful inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime is too old or the model cannot vend state.
    pub fn new_state(&self) -> Result<MLState, CoreMLError> {
        let mut error = ptr::null_mut();
        let mut state = ptr::null_mut();
        let status = unsafe { ffi::cm_model_new_state(self.ptr, &mut state, &mut error) };
        if status != ffi::status::OK || state.is_null() {
            return Err(from_swift(status, error));
        }
        MLState::from_raw(state)
            .ok_or_else(|| CoreMLError::StateFailed("CoreML returned no MLState".to_owned()))
    }

    /// Run a stateful prediction with default options.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML rejects the input features or the provided state.
    pub fn predict_with_state(
        &self,
        inputs: &FeatureProvider,
        state: &MLState,
    ) -> Result<FeatureProvider, CoreMLError> {
        let options = PredictionOptions::default();
        self.predict_with_state_and_options(inputs, state, &options)
    }

    /// Run a stateful prediction with explicit options.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML rejects the input features or the provided state.
    pub fn predict_with_state_and_options(
        &self,
        inputs: &FeatureProvider,
        state: &MLState,
        options: &PredictionOptions,
    ) -> Result<FeatureProvider, CoreMLError> {
        let options_json = options.as_json_c_string()?;
        let mut error = ptr::null_mut();
        let mut out = ptr::null_mut();
        let status = unsafe {
            ffi::cm_model_predict_with_state(
                self.ptr,
                inputs.ptr,
                state.ptr,
                options_json.as_ptr(),
                &mut out,
                &mut error,
            )
        };
        if status != ffi::status::OK || out.is_null() {
            return Err(from_swift(status, error));
        }
        FeatureProvider::from_raw(out).ok_or_else(|| {
            CoreMLError::StateFailed("CoreML stateful prediction returned no outputs".to_owned())
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
        CoreMLError::InvalidArgument(format!("path contains an interior NUL byte: {error}"))
    })
}
