use core::ffi::c_char;

extern "C" {
    pub fn cm_prediction_options_snapshot_json(
        prediction_options_json: *const c_char,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
}
