use core::ffi::c_char;

extern "C" {
    pub fn cm_compute_plan_load_summary(
        path: *const c_char,
        configuration_json: *const c_char,
        out_summary_json: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
}
