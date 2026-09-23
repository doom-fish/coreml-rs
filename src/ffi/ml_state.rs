use core::ffi::{c_char, c_void};

pub type StateMultiArrayCallback = unsafe extern "C" fn(buffer: *mut c_void, context: *mut c_void);

extern "C" {
    pub fn cm_state_runtime_supported() -> bool;
    pub fn cm_state_with_multi_array(
        state: *mut c_void,
        name: *const c_char,
        callback: StateMultiArrayCallback,
        context: *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
}
