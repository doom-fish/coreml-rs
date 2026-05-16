use core::ffi::c_char;

extern "C" {
    pub fn cm_model_compile(
        path: *const c_char,
        out_compiled_path: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
}
