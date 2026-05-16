//! Raw FFI declarations matching the Swift `@_cdecl` exports in
//! `swift-bridge/Sources/CoreMLBridge`.
//!
//! These functions are intentionally low-level and unsafe. Prefer the safe
//! wrappers in the parent modules.

use core::ffi::{c_char, c_void};

extern "C" {
    pub fn cm_object_release(ptr: *mut c_void);

    pub fn cm_model_load(
        path: *const c_char,
        compute_units: i32,
        allow_low_precision_accumulation_on_gpu: bool,
        model_display_name: *const c_char,
        out_model: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_load_from_specification(
        bytes: *const u8,
        byte_count: usize,
        compute_units: i32,
        allow_low_precision_accumulation_on_gpu: bool,
        model_display_name: *const c_char,
        out_model: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_compile(
        path: *const c_char,
        out_compiled_path: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_predict(
        model: *mut c_void,
        inputs: *mut c_void,
        out_provider: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_predict_batch(
        model: *mut c_void,
        batch: *mut c_void,
        out_batch: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_description_json(model: *mut c_void) -> *mut c_char;

    pub fn cm_multi_array_new(
        shape: *const i64,
        rank: usize,
        data_type: i32,
        out_array: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_multi_array_count(array: *mut c_void) -> usize;
    pub fn cm_multi_array_data_type(array: *mut c_void) -> i32;
    pub fn cm_multi_array_rank(array: *mut c_void) -> usize;
    pub fn cm_multi_array_copy_shape(
        array: *mut c_void,
        out_buffer: *mut i64,
        capacity: usize,
    ) -> usize;
    pub fn cm_multi_array_copy_strides(
        array: *mut c_void,
        out_buffer: *mut i64,
        capacity: usize,
    ) -> usize;
    pub fn cm_multi_array_data_pointer(array: *mut c_void) -> *mut c_void;

    pub fn cm_feature_provider_new() -> *mut c_void;
    pub fn cm_feature_provider_insert_multi_array(
        provider: *mut c_void,
        name: *const c_char,
        array: *mut c_void,
    ) -> i32;
    pub fn cm_feature_provider_insert_pixel_buffer(
        provider: *mut c_void,
        name: *const c_char,
        pixel_buffer: *mut c_void,
    ) -> i32;
    pub fn cm_feature_provider_insert_string(
        provider: *mut c_void,
        name: *const c_char,
        value: *const c_char,
    ) -> i32;
    pub fn cm_feature_provider_insert_int64(
        provider: *mut c_void,
        name: *const c_char,
        value: i64,
    ) -> i32;
    pub fn cm_feature_provider_insert_double(
        provider: *mut c_void,
        name: *const c_char,
        value: f64,
    ) -> i32;
    pub fn cm_feature_provider_keys_json(provider: *mut c_void) -> *mut c_char;
    pub fn cm_feature_provider_feature_type(provider: *mut c_void, name: *const c_char) -> i32;
    pub fn cm_feature_provider_get_multi_array(
        provider: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn cm_feature_provider_get_string(
        provider: *mut c_void,
        name: *const c_char,
    ) -> *mut c_char;
    pub fn cm_feature_provider_get_int64(
        provider: *mut c_void,
        name: *const c_char,
        out_value: *mut i64,
    ) -> bool;
    pub fn cm_feature_provider_get_double(
        provider: *mut c_void,
        name: *const c_char,
        out_value: *mut f64,
    ) -> bool;

    pub fn cm_batch_provider_new() -> *mut c_void;
    pub fn cm_batch_provider_push(batch: *mut c_void, provider: *mut c_void) -> i32;
    pub fn cm_batch_provider_count(batch: *mut c_void) -> usize;
    pub fn cm_batch_provider_get_provider(batch: *mut c_void, index: usize) -> *mut c_void;
}

pub mod status {
    pub const OK: i32 = 0;
    pub const INVALID_ARGUMENT: i32 = -1;
    pub const MODEL_LOAD_FAILED: i32 = -2;
    pub const PREDICTION_FAILED: i32 = -3;
    pub const COMPILATION_FAILED: i32 = -4;
    pub const FEATURE_PROVIDER_FAILED: i32 = -5;
    pub const MULTI_ARRAY_FAILED: i32 = -6;
    pub const UNSUPPORTED: i32 = -7;
    pub const TIMED_OUT: i32 = -8;
    pub const MODEL_ASSET_FAILED: i32 = -9;
    pub const INDEX_OUT_OF_RANGE: i32 = -10;
    pub const UNKNOWN: i32 = -99;
}
