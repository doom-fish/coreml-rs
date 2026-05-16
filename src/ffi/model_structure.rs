use core::ffi::c_char;

extern "C" {
    pub fn cm_model_structure_load_json(
        path: *const c_char,
        out_json: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
}
