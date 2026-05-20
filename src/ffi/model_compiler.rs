use core::ffi::{c_char, c_void};

pub type ModelCompilerAsyncCallback =
    extern "C" fn(status: i32, compiled_path: *mut c_char, error: *const c_char, user_data: *mut c_void);

extern "C" {
    pub fn cm_model_compile(
        path: *const c_char,
        out_compiled_path: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_compile_async(
        path: *const c_char,
        callback: ModelCompilerAsyncCallback,
        user_data: *mut c_void,
    );
}
