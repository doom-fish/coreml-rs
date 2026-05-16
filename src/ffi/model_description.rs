use core::ffi::{c_char, c_void};

extern "C" {
    pub fn cm_model_description_json(model: *mut c_void) -> *mut c_char;
}
