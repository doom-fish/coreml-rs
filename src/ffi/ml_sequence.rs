use core::ffi::{c_char, c_void};

extern "C" {
    pub fn cm_sequence_new_empty(feature_type: i32, error_out: *mut *mut c_char) -> *mut c_void;
    pub fn cm_sequence_new_strings(
        value_ptrs: *const *const c_char,
        len: usize,
        error_out: *mut *mut c_char,
    ) -> *mut c_void;
    pub fn cm_sequence_new_int64s(
        values: *const i64,
        len: usize,
        error_out: *mut *mut c_char,
    ) -> *mut c_void;
    pub fn cm_sequence_type(sequence: *mut c_void) -> i32;
    pub fn cm_sequence_get_strings_json(sequence: *mut c_void) -> *mut c_char;
    pub fn cm_sequence_get_int64s_json(sequence: *mut c_void) -> *mut c_char;
}
