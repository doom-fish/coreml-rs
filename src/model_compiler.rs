//! `.mlmodel` compilation helpers.

#[cfg(feature = "async")]
use core::ffi::{c_char, c_void};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr;

#[cfg(feature = "async")]
use doom_fish_utils::completion::{error_from_cstr, AsyncCompletion};
#[cfg(feature = "async")]
use doom_fish_utils::panic_safe::catch_user_panic;

use crate::configuration::ModelConfiguration;
use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::ffi;
#[cfg(feature = "async")]
use crate::model::decode_async_error;
#[cfg(feature = "async")]
use crate::model::encode_async_error;
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

    /// Compile a source `.mlmodel` file asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML cannot compile the source model.
    #[cfg(feature = "async")]
    #[cfg_attr(docsrs, doc(cfg(feature = "async")))]
    #[allow(clippy::future_not_send)]
    pub async fn compile_async(mlmodel_path: &Path) -> Result<PathBuf, CoreMLError> {
        let path = path_to_c_string(mlmodel_path)?;
        let (future, user_data) = AsyncCompletion::create();
        unsafe {
            ffi::cm_model_compile_async(path.as_ptr(), model_compile_async_callback, user_data);
        }
        future
            .await
            .map_err(|payload| decode_async_error(payload, ffi::status::COMPILATION_FAILED))
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

    /// Compile a source `.mlmodel` asynchronously and load the compiled bundle.
    ///
    /// # Errors
    ///
    /// Returns an error if compilation or loading fails.
    #[cfg(feature = "async")]
    #[cfg_attr(docsrs, doc(cfg(feature = "async")))]
    #[allow(clippy::future_not_send)]
    pub async fn compile_and_load_async(
        mlmodel_path: &Path,
        configuration: Option<&ModelConfiguration>,
    ) -> Result<Model, CoreMLError> {
        let compiled = Self::compile_async(mlmodel_path).await?;
        Model::load_async(compiled.as_path(), configuration).await
    }
}

#[cfg(feature = "async")]
extern "C" fn model_compile_async_callback(
    status: i32,
    compiled_path: *mut c_char,
    error: *const c_char,
    user_data: *mut c_void,
) {
    catch_user_panic("coreml::model_compile_async_callback", || {
        if status == ffi::status::OK {
            if compiled_path.is_null() {
                unsafe {
                    AsyncCompletion::<PathBuf>::complete_err(
                        user_data,
                        encode_async_error(
                            ffi::status::COMPILATION_FAILED,
                            "CoreML async compilation returned no compiled model path".to_owned(),
                        ),
                    );
                }
            } else {
                let path = PathBuf::from(take_owned_c_string(compiled_path));
                unsafe { AsyncCompletion::<PathBuf>::complete_ok(user_data, path) };
            }
            return;
        }

        if !compiled_path.is_null() {
            unsafe { libc::free(compiled_path.cast()) };
        }

        let message = unsafe { error_from_cstr(error) };
        unsafe {
            AsyncCompletion::<PathBuf>::complete_err(user_data, encode_async_error(status, message));
        }
    });
}

fn path_to_c_string(path: impl AsRef<Path>) -> Result<CString, CoreMLError> {
    CString::new(path.as_ref().to_string_lossy().into_owned()).map_err(|error| {
        CoreMLError::InvalidArgument(format!("path contains an interior NUL byte: {error}"))
    })
}
