use core::ffi::{c_char, c_void};

extern "C" {
    pub fn cm_feature_provider_new() -> *mut c_void;
    pub fn cm_feature_provider_insert_feature(
        provider: *mut c_void,
        name: *const c_char,
        feature: *mut c_void,
    ) -> i32;
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
    pub fn cm_feature_provider_get_feature(
        provider: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
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
}
