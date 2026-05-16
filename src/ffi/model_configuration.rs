use core::ffi::c_char;

extern "C" {
    pub fn cm_model_configuration_snapshot_json(
        configuration_json: *const c_char,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
}
