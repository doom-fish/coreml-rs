//! Rust callback registration for `MLCustomLayer` implementations.

use core::ffi::{c_char, c_void};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};

use doom_fish_utils::panic_safe::catch_user_panic;
use libc::strdup;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::ffi;
use crate::multi_array::MultiArrayRef;

/// Initialization context supplied when CoreML constructs a custom layer instance.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MLCustomLayerInitContext {
    /// Objective-C runtime class name registered with CoreML.
    pub class_name: String,
    /// JSON-safe contents of the model specification's `parameters` dictionary.
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
}

/// Trait implemented by Rust-backed `MLCustomLayer` instances.
pub trait MLCustomLayer: Send {
    /// Accept weight blobs loaded from the CoreML model specification.
    fn set_weight_data(&mut self, _weights: &[Vec<u8>]) -> Result<(), CoreMLError> {
        Ok(())
    }

    /// Compute the output shapes for the provided input shapes.
    fn output_shapes_for_input_shapes(
        &self,
        input_shapes: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>, CoreMLError>;

    /// Execute the layer on CPU-backed `MLMultiArray` values.
    fn evaluate_on_cpu(
        &mut self,
        inputs: &[&MultiArrayRef],
        outputs: &mut [&mut MultiArrayRef],
    ) -> Result<(), CoreMLError>;

    /// Optional GPU callback receiving raw Metal object pointers.
    ///
    /// # Safety
    ///
    /// The command buffer and texture pointers are borrowed Objective-C / Metal objects that are
    /// only valid for the duration of this callback.
    unsafe fn encode_to_command_buffer(
        &mut self,
        _command_buffer: *mut c_void,
        _inputs: &[*mut c_void],
        _outputs: &[*mut c_void],
    ) -> Result<(), CoreMLError> {
        Err(CoreMLError::Unsupported(
            "this MLCustomLayer registration does not support GPU command-buffer encoding"
                .to_owned(),
        ))
    }

    /// Whether the optional GPU callback should be surfaced to CoreML.
    fn supports_gpu_encoding(&self) -> bool {
        false
    }
}

type LayerFactory =
    dyn Fn(MLCustomLayerInitContext) -> Result<Box<dyn MLCustomLayer>, CoreMLError> + Send + Sync;

struct LayerInstanceBox {
    inner: Mutex<Box<dyn MLCustomLayer>>,
}

/// Lifetime guard for one registered Objective-C custom-layer class.
pub struct MLCustomLayerRegistration {
    class_name: String,
}

/// Retained handle that can manually exercise a registered `MLCustomLayer`.
pub struct MLCustomLayerHandle {
    class_name: String,
    ptr: *mut c_void,
}

impl MLCustomLayerRegistration {
    /// Register a Rust factory under the supplied Objective-C runtime class name.
    ///
    /// Keep the returned registration alive while CoreML may need to instantiate the layer.
    ///
    /// # Errors
    ///
    /// Returns an error if the class name is invalid, collides with an incompatible Objective-C
    /// class, or has already been registered in Rust.
    pub fn register<L, F>(class_name: impl Into<String>, factory: F) -> Result<Self, CoreMLError>
    where
        L: MLCustomLayer + 'static,
        F: Fn(MLCustomLayerInitContext) -> Result<L, CoreMLError> + Send + Sync + 'static,
    {
        let class_name = class_name.into();
        let class_name_c = c_string(&class_name, "custom layer class name")?;
        let erased: Arc<LayerFactory> = Arc::new(move |context| {
            factory(context).map(|layer| Box::new(layer) as Box<dyn MLCustomLayer>)
        });

        let mut error = ptr::null_mut();
        let status =
            unsafe { ffi::cm_custom_layer_register_class(class_name_c.as_ptr(), &raw mut error) };
        if status != ffi::status::OK {
            return Err(from_swift(status, error));
        }

        let mut registry = layer_registry().write().map_err(|_| {
            CoreMLError::CustomLayerFailed("custom-layer registry lock was poisoned".to_owned())
        })?;
        if registry.contains_key(&class_name) {
            return Err(CoreMLError::InvalidArgument(format!(
                "custom layer class '{class_name}' is already registered"
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

    /// Manually instantiate the registered layer with a JSON-safe parameters dictionary.
    ///
    /// This is primarily useful for headless tests/examples; real CoreML model loading uses the
    /// same registration but constructs instances itself.
    ///
    /// # Errors
    ///
    /// Returns an error if the Swift bridge cannot instantiate the registered class.
    pub fn instantiate(
        &self,
        parameters: &BTreeMap<String, Value>,
    ) -> Result<MLCustomLayerHandle, CoreMLError> {
        let class_name = c_string(&self.class_name, "custom layer class name")?;
        let parameters_json = json_c_string(parameters, "custom layer parameters")?;
        let mut error = ptr::null_mut();
        let mut layer = ptr::null_mut();
        let status = unsafe {
            ffi::cm_custom_layer_create(
                class_name.as_ptr(),
                parameters_json.as_ptr(),
                &raw mut layer,
                &raw mut error,
            )
        };
        if status != ffi::status::OK || layer.is_null() {
            return Err(from_swift(status, error));
        }
        Ok(MLCustomLayerHandle {
            class_name: self.class_name.clone(),
            ptr: layer,
        })
    }
}

impl Drop for MLCustomLayerRegistration {
    fn drop(&mut self) {
        if let Ok(mut registry) = layer_registry().write() {
            registry.remove(&self.class_name);
        }
    }
}

impl core::fmt::Debug for MLCustomLayerRegistration {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MLCustomLayerRegistration")
            .field("class_name", &self.class_name)
            .finish()
    }
}

impl MLCustomLayerHandle {
    /// Objective-C runtime class name backing this instance.
    #[must_use]
    pub fn class_name(&self) -> &str {
        &self.class_name
    }

    /// Forward weight blobs to the registered Rust callback.
    ///
    /// # Errors
    ///
    /// Returns an error if the Swift bridge or Rust callback rejects the weights.
    pub fn set_weight_data(&mut self, weights: &[Vec<u8>]) -> Result<(), CoreMLError> {
        let weight_ptrs: Vec<*const u8> = weights
            .iter()
            .map(|weight| {
                if weight.is_empty() {
                    ptr::null()
                } else {
                    weight.as_ptr()
                }
            })
            .collect();
        let weight_lengths: Vec<usize> = weights.iter().map(Vec::len).collect();
        let mut error = ptr::null_mut();
        let status = unsafe {
            ffi::cm_custom_layer_set_weight_data(
                self.ptr,
                weight_ptrs.as_ptr(),
                weight_lengths.as_ptr(),
                weights.len(),
                &raw mut error,
            )
        };
        if status != ffi::status::OK {
            return Err(from_swift(status, error));
        }
        Ok(())
    }

    /// Ask the layer for its output shapes.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization fails or the Rust callback rejects the shapes.
    pub fn output_shapes_for_input_shapes(
        &self,
        input_shapes: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>, CoreMLError> {
        let input_shapes_json = json_c_string(input_shapes, "custom layer input shapes")?;
        let mut error = ptr::null_mut();
        let mut output_json = ptr::null_mut();
        let status = unsafe {
            ffi::cm_custom_layer_output_shapes_json(
                self.ptr,
                input_shapes_json.as_ptr(),
                &raw mut output_json,
                &raw mut error,
            )
        };
        if status != ffi::status::OK || output_json.is_null() {
            return Err(from_swift(status, error));
        }
        serde_json::from_str::<Vec<Vec<usize>>>(&take_owned_c_string(output_json)).map_err(
            |decode_error| {
                CoreMLError::CustomLayerFailed(format!(
                    "failed to decode custom-layer output-shape JSON: {decode_error}"
                ))
            },
        )
    }

    /// Run the layer's CPU callback against existing `MLMultiArray` values.
    ///
    /// # Errors
    ///
    /// Returns an error if the bridge rejects the arrays or the Rust callback fails.
    pub fn evaluate_on_cpu(
        &mut self,
        inputs: &[&MultiArrayRef],
        outputs: &mut [&mut MultiArrayRef],
    ) -> Result<(), CoreMLError> {
        let input_ptrs: Vec<*mut c_void> = inputs.iter().map(|input| input.as_ptr()).collect();
        let output_ptrs: Vec<*mut c_void> = outputs.iter().map(|output| output.as_ptr()).collect();
        let mut error = ptr::null_mut();
        let status = unsafe {
            ffi::cm_custom_layer_evaluate_cpu(
                self.ptr,
                input_ptrs.as_ptr(),
                input_ptrs.len(),
                output_ptrs.as_ptr(),
                output_ptrs.len(),
                &raw mut error,
            )
        };
        if status != ffi::status::OK {
            return Err(from_swift(status, error));
        }
        Ok(())
    }
}

impl Drop for MLCustomLayerHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cm_object_release(self.ptr) };
        }
    }
}

impl core::fmt::Debug for MLCustomLayerHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MLCustomLayerHandle")
            .field("class_name", &self.class_name)
            .finish_non_exhaustive()
    }
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn cm_rust_custom_layer_create(
    class_name: *const c_char,
    parameters_json: *const c_char,
    out_context: *mut *mut c_void,
    out_flags: *mut u32,
    error_out: *mut *mut c_char,
) -> i32 {
    ffi_callback(error_out, || {
        if out_context.is_null() || out_flags.is_null() {
            return Err(CoreMLError::InvalidArgument(
                "custom layer callback expected non-null out pointers".to_owned(),
            ));
        }
        *out_context = ptr::null_mut();
        *out_flags = 0;

        let class_name = required_string(class_name, "custom layer class name")?;
        let parameters = parse_json::<BTreeMap<String, Value>>(parameters_json)?;
        let context = MLCustomLayerInitContext {
            class_name: class_name.clone(),
            parameters,
        };
        let factory = lookup_factory(&class_name)?;
        let layer = factory(context)?;
        let flags = u32::from(layer.supports_gpu_encoding());
        *out_context = Box::into_raw(Box::new(LayerInstanceBox {
            inner: Mutex::new(layer),
        }))
        .cast();
        *out_flags = flags;
        Ok(())
    })
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn cm_rust_custom_layer_set_weight_data(
    context: *mut c_void,
    weight_data: *const *const u8,
    weight_lengths: *const usize,
    weight_count: usize,
    error_out: *mut *mut c_char,
) -> i32 {
    ffi_callback(error_out, || {
        let weights = byte_vectors(weight_data, weight_lengths, weight_count)?;
        lock_layer(context)?.set_weight_data(&weights)
    })
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn cm_rust_custom_layer_output_shapes_json(
    context: *mut c_void,
    input_shapes_json: *const c_char,
    out_json: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> i32 {
    ffi_callback(error_out, || {
        if out_json.is_null() {
            return Err(CoreMLError::InvalidArgument(
                "custom layer output-shape callback expected a non-null JSON pointer".to_owned(),
            ));
        }
        *out_json = ptr::null_mut();

        let input_shapes = parse_json::<Vec<Vec<usize>>>(input_shapes_json)?;
        let output_shapes = lock_layer(context)?.output_shapes_for_input_shapes(&input_shapes)?;
        let json = serde_json::to_string(&output_shapes).map_err(|error| {
            CoreMLError::CustomLayerFailed(format!(
                "failed to encode custom-layer output shapes: {error}"
            ))
        })?;
        *out_json = duplicate_c_string(&json);
        Ok(())
    })
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn cm_rust_custom_layer_evaluate_cpu(
    context: *mut c_void,
    inputs: *const *mut c_void,
    input_count: usize,
    outputs: *const *mut c_void,
    output_count: usize,
    error_out: *mut *mut c_char,
) -> i32 {
    ffi_callback(error_out, || {
        let input_ptrs = non_null_pointers(inputs, input_count, "custom-layer input")?;
        let output_ptrs = non_null_pointers(outputs, output_count, "custom-layer output")?;
        reject_aliased_arrays(&input_ptrs, &output_ptrs)?;
        let input_arrays: Vec<&MultiArrayRef> = input_ptrs
            .iter()
            .map(|&ptr| unsafe { MultiArrayRef::from_raw(ptr) })
            .collect();
        let mut output_arrays: Vec<&mut MultiArrayRef> = output_ptrs
            .iter()
            .map(|&ptr| unsafe { MultiArrayRef::from_raw_mut(ptr) })
            .collect();
        lock_layer(context)?.evaluate_on_cpu(&input_arrays, &mut output_arrays)
    })
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn cm_rust_custom_layer_encode(
    context: *mut c_void,
    command_buffer: *mut c_void,
    inputs: *const *mut c_void,
    input_count: usize,
    outputs: *const *mut c_void,
    output_count: usize,
    error_out: *mut *mut c_char,
) -> i32 {
    ffi_callback(error_out, || {
        if command_buffer.is_null() {
            return Err(CoreMLError::InvalidArgument(
                "custom layer GPU encoding requires a non-null command buffer".to_owned(),
            ));
        }
        let input_ptrs = collect_raw_pointers(inputs, input_count, "custom-layer GPU inputs")?;
        let output_ptrs = collect_raw_pointers(outputs, output_count, "custom-layer GPU outputs")?;
        let mut layer = lock_layer(context)?;
        unsafe { layer.encode_to_command_buffer(command_buffer, &input_ptrs, &output_ptrs) }
    })
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn cm_rust_custom_layer_release(context: *mut c_void) {
    if !context.is_null() {
        catch_user_panic("coreml::custom_layer_release", || {
            drop(unsafe { Box::from_raw(context.cast::<LayerInstanceBox>()) });
        });
    }
}

fn layer_registry() -> &'static RwLock<BTreeMap<String, Arc<LayerFactory>>> {
    static REGISTRY: OnceLock<RwLock<BTreeMap<String, Arc<LayerFactory>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(BTreeMap::new()))
}

fn lookup_factory(class_name: &str) -> Result<Arc<LayerFactory>, CoreMLError> {
    layer_registry()
        .read()
        .map_err(|_| {
            CoreMLError::CustomLayerFailed("custom-layer registry lock was poisoned".to_owned())
        })?
        .get(class_name)
        .cloned()
        .ok_or_else(|| {
            CoreMLError::InvalidArgument(format!(
                "no Rust MLCustomLayer factory is registered for class '{class_name}'"
            ))
        })
}

unsafe fn lock_layer<'a>(
    context: *mut c_void,
) -> Result<MutexGuard<'a, Box<dyn MLCustomLayer>>, CoreMLError> {
    let Some(instance) = (unsafe { context.cast::<LayerInstanceBox>().as_ref() }) else {
        return Err(CoreMLError::InvalidArgument(
            "custom-layer callback received a null instance pointer".to_owned(),
        ));
    };
    instance.inner.lock().map_err(|_| {
        CoreMLError::CustomLayerFailed(
            "custom-layer instance is unusable after an earlier callback panicked".to_owned(),
        )
    })
}

fn c_string(value: &str, label: &str) -> Result<CString, CoreMLError> {
    CString::new(value).map_err(|error| {
        CoreMLError::InvalidArgument(format!("{label} contains an interior NUL byte: {error}"))
    })
}

fn json_c_string<T: ?Sized + Serialize>(value: &T, label: &str) -> Result<CString, CoreMLError> {
    CString::new(serde_json::to_string(value).map_err(|error| {
        CoreMLError::CustomLayerFailed(format!("failed to encode {label} as JSON: {error}"))
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
            "failed to decode custom-layer JSON payload: {error}"
        ))
    })
}

fn byte_vectors(
    data_ptrs: *const *const u8,
    lengths: *const usize,
    count: usize,
) -> Result<Vec<Vec<u8>>, CoreMLError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if data_ptrs.is_null() || lengths.is_null() {
        return Err(CoreMLError::InvalidArgument(
            "custom-layer weights require non-null data and length arrays".to_owned(),
        ));
    }
    let ptrs = unsafe { std::slice::from_raw_parts(data_ptrs, count) };
    let lens = unsafe { std::slice::from_raw_parts(lengths, count) };
    ptrs.iter()
        .zip(lens)
        .enumerate()
        .map(|(index, (&data_ptr, &len))| {
            if len == 0 {
                return Ok(Vec::new());
            }
            if data_ptr.is_null() {
                return Err(CoreMLError::InvalidArgument(format!(
                    "custom-layer weight blob {index} was null despite having a non-zero length"
                )));
            }
            Ok(unsafe { std::slice::from_raw_parts(data_ptr, len) }.to_vec())
        })
        .collect()
}

fn collect_raw_pointers(
    values: *const *mut c_void,
    count: usize,
    label: &str,
) -> Result<Vec<*mut c_void>, CoreMLError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if values.is_null() {
        return Err(CoreMLError::InvalidArgument(format!(
            "{label} must not be null when count is non-zero"
        )));
    }
    Ok(unsafe { std::slice::from_raw_parts(values, count) }.to_vec())
}

fn non_null_pointers(
    values: *const *mut c_void,
    count: usize,
    label: &str,
) -> Result<Vec<NonNull<c_void>>, CoreMLError> {
    collect_raw_pointers(values, count, label)?
        .into_iter()
        .enumerate()
        .map(|(index, ptr)| {
            NonNull::new(ptr).ok_or_else(|| {
                CoreMLError::InvalidArgument(format!("{label} {index} must not be null"))
            })
        })
        .collect()
}

fn reject_aliased_arrays(
    inputs: &[NonNull<c_void>],
    outputs: &[NonNull<c_void>],
) -> Result<(), CoreMLError> {
    for (index, output) in outputs.iter().enumerate() {
        if outputs[..index].contains(output) || inputs.contains(output) {
            return Err(CoreMLError::CustomLayerFailed(format!(
                "custom-layer output {index} aliases another input or output array"
            )));
        }
    }
    Ok(())
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
            write_error(error_out, "panic in Rust MLCustomLayer callback");
            ffi::status::CUSTOM_LAYER_FAILED
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
        CString::new("failed to encode custom-layer error message")
            .expect("static strings are valid")
    });
    unsafe { strdup(c_string.as_ptr()) }
}

#[cfg(test)]
mod tests {
    use core::ffi::c_void;
    use std::ptr::NonNull;

    use super::reject_aliased_arrays;

    #[test]
    fn aliased_outputs_are_rejected() {
        let slots = [0_u8; 3];
        let pointer = |index: usize| NonNull::from(&slots[index]).cast::<c_void>();
        assert!(reject_aliased_arrays(&[pointer(0)], &[pointer(1), pointer(2)]).is_ok());
        assert!(reject_aliased_arrays(&[pointer(0), pointer(0)], &[pointer(1)]).is_ok());
        assert!(reject_aliased_arrays(&[pointer(0)], &[pointer(0)]).is_err());
        assert!(reject_aliased_arrays(&[], &[pointer(1), pointer(1)]).is_err());
    }
}
