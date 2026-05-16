use core::ffi::c_char;

extern "C" {
    pub fn cm_all_compute_devices_json(
        out_json: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
}
