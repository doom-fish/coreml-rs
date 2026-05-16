//! `.mlmodel` compilation helpers.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr;

use crate::configuration::ModelConfiguration;
use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::ffi;
use crate::model::Model;

/// Utilities mirroring `MLModel`'s model-compilation APIs.
#[derive(Debug, Default, Clone, Copy)]
pub struct ModelCompiler;

impl ModelCompiler {
    /// Compile a source `.mlmodel` file to a temporary `.mlmodelc` bundle.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML cannot compile the source model.
    pub fn compile(mlmodel_path: impl AsRef<Path>) -> Result<PathBuf, CoreMLError> {
        let path = path_to_c_string(mlmodel_path)?;
        let mut error = ptr::null_mut();
        let mut compiled_path = ptr::null_mut();
        let status =
            unsafe { ffi::cm_model_compile(path.as_ptr(), &mut compiled_path, &mut error) };
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
    ) -> Result<Model, CoreMLError> {
        let compiled = Self::compile(mlmodel_path)?;
        Model::load_from_url(compiled, configuration)
    }
}

fn path_to_c_string(path: impl AsRef<Path>) -> Result<CString, CoreMLError> {
    CString::new(path.as_ref().to_string_lossy().into_owned()).map_err(|error| {
        CoreMLError::InvalidArgument(format!("path contains an interior NUL byte: {error}"))
    })
}
