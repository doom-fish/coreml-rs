use core::ffi::{c_char, c_void};

extern "C" {
    pub fn cm_update_run(
        model_path: *const c_char,
        training_data: *mut c_void,
        configuration_json: *const c_char,
        handlers_json: *const c_char,
        out_result_json: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
}
