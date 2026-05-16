use core::ffi::{c_char, c_void};

extern "C" {
    pub fn cm_state_runtime_supported() -> bool;
    pub fn cm_state_snapshot_multi_array(
        state: *mut c_void,
        name: *const c_char,
        out_array: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
}
