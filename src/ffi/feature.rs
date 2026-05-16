use core::ffi::{c_char, c_void};

extern "C" {
    pub fn cm_feature_new_int64(value: i64) -> *mut c_void;
    pub fn cm_feature_new_double(value: f64) -> *mut c_void;
    pub fn cm_feature_new_string(value: *const c_char) -> *mut c_void;
    pub fn cm_feature_new_multi_array(array: *mut c_void) -> *mut c_void;
    pub fn cm_feature_new_undefined(feature_type: i32) -> *mut c_void;
    pub fn cm_feature_new_string_dictionary(
        keys: *const *const c_char,
        values: *const f64,
        len: usize,
        error_out: *mut *mut c_char,
    ) -> *mut c_void;
    pub fn cm_feature_new_int64_dictionary(
        keys: *const i64,
        values: *const f64,
        len: usize,
        error_out: *mut *mut c_char,
    ) -> *mut c_void;
    pub fn cm_feature_type(feature: *mut c_void) -> i32;
    pub fn cm_feature_is_undefined(feature: *mut c_void) -> bool;
    pub fn cm_feature_get_int64(feature: *mut c_void, out_value: *mut i64) -> bool;
    pub fn cm_feature_get_double(feature: *mut c_void, out_value: *mut f64) -> bool;
    pub fn cm_feature_get_string(feature: *mut c_void) -> *mut c_char;
    pub fn cm_feature_get_multi_array(feature: *mut c_void) -> *mut c_void;
    pub fn cm_feature_get_string_dictionary_json(feature: *mut c_void) -> *mut c_char;
    pub fn cm_feature_get_int64_dictionary_json(feature: *mut c_void) -> *mut c_char;
    pub fn cm_feature_is_equal(lhs: *mut c_void, rhs: *mut c_void) -> bool;
}
