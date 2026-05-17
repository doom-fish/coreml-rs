//! Rust callback registration for `MLCustomModel` implementations.

use core::ffi::{c_char, c_void};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::{Arc, OnceLock, RwLock};

use libc::strdup;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{from_status_message, from_swift, CoreMLError};
use crate::feature_provider::{BatchProvider, FeatureProvider};
use crate::ffi;
use crate::model_description::ModelDescription;
use crate::prediction::PredictionOptions;

/// Initialization context supplied when CoreML constructs a custom model instance.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MLCustomModelInitContext {
    /// Objective-C runtime class name registered with CoreML.
    pub class_name: String,
    /// Snapshot of the CoreML model description for the custom model.
    #[serde(default)]
    pub model_description: ModelDescription,
    /// JSON-safe contents of the model specification's `parameters` dictionary.
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
}

/// Trait implemented by Rust-backed `MLCustomModel` instances.
pub trait MLCustomModel: Send {
    /// Execute one single-sample prediction.
    fn prediction_from_features(
        &mut self,
        input: &FeatureProvider,
        options: &PredictionOptions,
    ) -> Result<FeatureProvider, CoreMLError>;

    /// Execute a batch prediction.
    fn predictions_from_batch(
        &mut self,
        input_batch: &BatchProvider,
        options: &PredictionOptions,
    ) -> Result<BatchProvider, CoreMLError> {
        let mut outputs = BatchProvider::new();
        for index in 0..input_batch.len() {
            let input = input_batch.get(index).ok_or_else(|| {
                CoreMLError::CustomModelFailed(format!(
                    "custom-model batch input {index} disappeared before prediction"
                ))
            })?;
            let output = self.prediction_from_features(&input, options)?;
            outputs.try_push(output)?;
        }
        Ok(outputs)
    }
}

type ModelFactory =
    dyn Fn(MLCustomModelInitContext) -> Result<Box<dyn MLCustomModel>, CoreMLError> + Send + Sync;

struct ModelInstanceBox {
    inner: Box<dyn MLCustomModel>,
}

/// Lifetime guard for one registered Objective-C custom-model class.
pub struct MLCustomModelRegistration {
    class_name: String,
}

/// Retained handle that can manually exercise a registered `MLCustomModel`.
pub struct MLCustomModelHandle {
    class_name: String,
    ptr: *mut c_void,
}

impl MLCustomModelRegistration {
    /// Register a Rust factory under the supplied Objective-C runtime class name.
    ///
    /// Keep the returned registration alive while CoreML may need to instantiate the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the class name is invalid, collides with an incompatible Objective-C
    /// class, or has already been registered in Rust.
    pub fn register<M, F>(class_name: impl Into<String>, factory: F) -> Result<Self, CoreMLError>
    where
        M: MLCustomModel + 'static,
        F: Fn(MLCustomModelInitContext) -> Result<M, CoreMLError> + Send + Sync + 'static,
    {
        let class_name = class_name.into();
        let class_name_c = c_string(&class_name, "custom model class name")?;
        let erased: Arc<ModelFactory> = Arc::new(move |context| {
            factory(context).map(|model| Box::new(model) as Box<dyn MLCustomModel>)
        });

        let mut error = ptr::null_mut();
        let status =
            unsafe { ffi::cm_custom_model_register_class(class_name_c.as_ptr(), &mut error) };
        if status != ffi::status::OK {
            return Err(from_swift(status, error));
        }

        let mut registry = model_registry().write().map_err(|_| {
            CoreMLError::CustomModelFailed("custom-model registry lock was poisoned".to_owned())
        })?;
        if registry.contains_key(&class_name) {
            return Err(CoreMLError::InvalidArgument(format!(
                "custom model class '{class_name}' is already registered"
            )));
        }
        registry.insert(class_name.clone(), erased);
        drop(registry);
        Ok(Self { class_name })
    }

    /// Objective-C runtime class name passed to CoreML.
    #[must_use]
    pub fn class_name(&self) -> &str {
        &self.class_name
    }

    /// Manually instantiate the registered model with a JSON-safe parameters dictionary.
    ///
    /// This test harness uses an empty `MLModelDescription()` instance. Real CoreML model loading
    /// uses the same registration but supplies the true model description.
    ///
    /// # Errors
    ///
    /// Returns an error if the Swift bridge cannot instantiate the registered class.
    pub fn instantiate(
        &self,
        parameters: &BTreeMap<String, Value>,
    ) -> Result<MLCustomModelHandle, CoreMLError> {
        let class_name = c_string(&self.class_name, "custom model class name")?;
        let parameters_json = json_c_string(parameters, "custom model parameters")?;
        let mut error = ptr::null_mut();
        let mut model = ptr::null_mut();
        let status = unsafe {
            ffi::cm_custom_model_create(
                class_name.as_ptr(),
                parameters_json.as_ptr(),
                &mut model,
                &mut error,
            )
        };
        if status != ffi::status::OK || model.is_null() {
            return Err(from_swift(status, error));
        }
        Ok(MLCustomModelHandle {
            class_name: self.class_name.clone(),
            ptr: model,
        })
    }
}

impl Drop for MLCustomModelRegistration {
    fn drop(&mut self) {
        if let Ok(mut registry) = model_registry().write() {
            registry.remove(&self.class_name);
        }
    }
}

impl core::fmt::Debug for MLCustomModelRegistration {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MLCustomModelRegistration")
            .field("class_name", &self.class_name)
            .finish()
    }
}

impl MLCustomModelHandle {
    /// Objective-C runtime class name backing this instance.
    #[must_use]
    pub fn class_name(&self) -> &str {
        &self.class_name
    }

    /// Execute one single-sample prediction.
    ///
    /// # Errors
    ///
    /// Returns an error if the Swift bridge or Rust callback rejects the request.
    pub fn predict(
        &mut self,
        input: &FeatureProvider,
        options: &PredictionOptions,
    ) -> Result<FeatureProvider, CoreMLError> {
        let options_json = options.as_json_c_string()?;
        let mut error = ptr::null_mut();
        let mut output = ptr::null_mut();
        let status = unsafe {
            ffi::cm_custom_model_predict(
                self.ptr,
                input.ptr,
                options_json.as_ptr(),
                &mut output,
                &mut error,
            )
        };
        if status != ffi::status::OK || output.is_null() {
            return Err(from_swift(status, error));
        }
        FeatureProvider::from_raw(output).ok_or_else(|| {
            from_status_message(
                ffi::status::CUSTOM_MODEL_FAILED,
                "custom-model prediction returned no feature provider".to_owned(),
            )
        })
    }

    /// Execute one batch prediction.
    ///
    /// # Errors
    ///
    /// Returns an error if the Swift bridge or Rust callback rejects the request.
    pub fn predict_batch(
        &mut self,
        input_batch: &BatchProvider,
        options: &PredictionOptions,
    ) -> Result<BatchProvider, CoreMLError> {
        let options_json = options.as_json_c_string()?;
        let mut error = ptr::null_mut();
        let mut output = ptr::null_mut();
        let status = unsafe {
            ffi::cm_custom_model_predict_batch(
                self.ptr,
                input_batch.ptr,
                options_json.as_ptr(),
                &mut output,
                &mut error,
            )
        };
        if status != ffi::status::OK || output.is_null() {
            return Err(from_swift(status, error));
        }
        BatchProvider::from_raw(output).ok_or_else(|| {
            from_status_message(
                ffi::status::CUSTOM_MODEL_FAILED,
                "custom-model batch prediction returned no batch provider".to_owned(),
            )
        })
    }
}

impl Drop for MLCustomModelHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cm_object_release(self.ptr) };
        }
    }
}

impl core::fmt::Debug for MLCustomModelHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MLCustomModelHandle")
            .field("class_name", &self.class_name)
            .finish_non_exhaustive()
    }
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn cm_rust_custom_model_create(
    class_name: *const c_char,
    model_description_json: *const c_char,
    parameters_json: *const c_char,
    out_context: *mut *mut c_void,
    error_out: *mut *mut c_char,
) -> i32 {
    ffi_callback(error_out, || {
        if out_context.is_null() {
            return Err(CoreMLError::InvalidArgument(
                "custom model callback expected a non-null out pointer".to_owned(),
            ));
        }
        *out_context = ptr::null_mut();

        let class_name = required_string(class_name, "custom model class name")?;
        let model_description = parse_json::<ModelDescription>(model_description_json)?;
        let parameters = parse_json::<BTreeMap<String, Value>>(parameters_json)?;
        let context = MLCustomModelInitContext {
            class_name: class_name.clone(),
            model_description,
            parameters,
        };
        let factory = lookup_factory(&class_name)?;
        let model = factory(context)?;
        *out_context = Box::into_raw(Box::new(ModelInstanceBox { inner: model })).cast();
        Ok(())
    })
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn cm_rust_custom_model_predict(
    context: *mut c_void,
    input: *mut c_void,
    prediction_options_json: *const c_char,
    out_provider: *mut *mut c_void,
    error_out: *mut *mut c_char,
) -> i32 {
    ffi_callback(error_out, || {
        if out_provider.is_null() {
            return Err(CoreMLError::InvalidArgument(
                "custom-model prediction expected a non-null output pointer".to_owned(),
            ));
        }
        *out_provider = ptr::null_mut();

        let model = model_from_ptr(context)?;
        let input = FeatureProvider::from_raw(input).ok_or_else(|| {
            CoreMLError::InvalidArgument(
                "custom-model prediction received a null feature-provider pointer".to_owned(),
            )
        })?;
        let options = parse_json::<PredictionOptions>(prediction_options_json)?;
        let output = model.inner.prediction_from_features(&input, &options)?;
        *out_provider = output.ptr;
        std::mem::forget(output);
        Ok(())
    })
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn cm_rust_custom_model_predict_batch(
    context: *mut c_void,
    batch: *mut c_void,
    prediction_options_json: *const c_char,
    out_batch: *mut *mut c_void,
    error_out: *mut *mut c_char,
) -> i32 {
    ffi_callback(error_out, || {
        if out_batch.is_null() {
            return Err(CoreMLError::InvalidArgument(
                "custom-model batch prediction expected a non-null output pointer".to_owned(),
            ));
        }
        *out_batch = ptr::null_mut();

        let model = model_from_ptr(context)?;
        let batch = BatchProvider::from_raw(batch).ok_or_else(|| {
            CoreMLError::InvalidArgument(
                "custom-model batch prediction received a null batch-provider pointer".to_owned(),
            )
        })?;
        let options = parse_json::<PredictionOptions>(prediction_options_json)?;
        let output = model.inner.predictions_from_batch(&batch, &options)?;
        *out_batch = output.ptr;
        std::mem::forget(output);
        Ok(())
    })
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn cm_rust_custom_model_release(context: *mut c_void) {
    if !context.is_null() {
        drop(Box::from_raw(context.cast::<ModelInstanceBox>()));
    }
}

fn model_registry() -> &'static RwLock<BTreeMap<String, Arc<ModelFactory>>> {
    static REGISTRY: OnceLock<RwLock<BTreeMap<String, Arc<ModelFactory>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(BTreeMap::new()))
}

fn lookup_factory(class_name: &str) -> Result<Arc<ModelFactory>, CoreMLError> {
    model_registry()
        .read()
        .map_err(|_| {
            CoreMLError::CustomModelFailed("custom-model registry lock was poisoned".to_owned())
        })?
        .get(class_name)
        .cloned()
        .ok_or_else(|| {
            CoreMLError::InvalidArgument(format!(
                "no Rust MLCustomModel factory is registered for class '{class_name}'"
            ))
        })
}

unsafe fn model_from_ptr<'a>(
    context: *mut c_void,
) -> Result<&'a mut ModelInstanceBox, CoreMLError> {
    if context.is_null() {
        return Err(CoreMLError::InvalidArgument(
            "custom-model callback received a null instance pointer".to_owned(),
        ));
    }
    Ok(&mut *context.cast::<ModelInstanceBox>())
}

fn c_string(value: &str, label: &str) -> Result<CString, CoreMLError> {
    CString::new(value).map_err(|error| {
        CoreMLError::InvalidArgument(format!("{label} contains an interior NUL byte: {error}"))
    })
}

fn json_c_string<T: ?Sized + Serialize>(value: &T, label: &str) -> Result<CString, CoreMLError> {
    CString::new(serde_json::to_string(value).map_err(|error| {
        CoreMLError::CustomModelFailed(format!("failed to encode {label} as JSON: {error}"))
    })?)
    .map_err(|error| {
        CoreMLError::InvalidArgument(format!(
            "{label} JSON contained an interior NUL byte: {error}"
        ))
    })
}

fn required_string(ptr: *const c_char, label: &str) -> Result<String, CoreMLError> {
    if ptr.is_null() {
        return Err(CoreMLError::InvalidArgument(format!(
            "{label} must not be null"
        )));
    }
    Ok(unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned())
}

fn parse_json<T>(ptr: *const c_char) -> Result<T, CoreMLError>
where
    T: DeserializeOwned + Default,
{
    if ptr.is_null() {
        return Ok(T::default());
    }
    let json = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
    if json.trim().is_empty() {
        return Ok(T::default());
    }
    serde_json::from_str(&json).map_err(|error| {
        CoreMLError::InvalidArgument(format!(
            "failed to decode custom-model JSON payload: {error}"
        ))
    })
}

fn ffi_callback(
    error_out: *mut *mut c_char,
    callback: impl FnOnce() -> Result<(), CoreMLError>,
) -> i32 {
    if !error_out.is_null() {
        unsafe { *error_out = ptr::null_mut() };
    }
    match catch_unwind(AssertUnwindSafe(callback)) {
        Ok(Ok(())) => ffi::status::OK,
        Ok(Err(error)) => {
            write_error(error_out, error.message());
            error.code()
        }
        Err(_) => {
            write_error(error_out, "panic in Rust MLCustomModel callback");
            ffi::status::CUSTOM_MODEL_FAILED
        }
    }
}

fn write_error(error_out: *mut *mut c_char, message: &str) {
    if error_out.is_null() {
        return;
    }
    unsafe { *error_out = duplicate_c_string(message) };
}

fn duplicate_c_string(message: &str) -> *mut c_char {
    let sanitized = message.replace('\0', " ");
    let c_string = CString::new(sanitized).unwrap_or_else(|_| {
        CString::new("failed to encode custom-model error message")
            .expect("static strings are valid")
    });
    unsafe { strdup(c_string.as_ptr()) }
}
