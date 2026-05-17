use core::ffi::{c_char, c_void};

extern "C" {
    pub fn cm_custom_model_register_class(
        class_name: *const c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_custom_model_create(
        class_name: *const c_char,
        parameters_json: *const c_char,
        out_model: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_custom_model_predict(
        model: *mut c_void,
        input: *mut c_void,
        prediction_options_json: *const c_char,
        out_provider: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_custom_model_predict_batch(
        model: *mut c_void,
        batch: *mut c_void,
        prediction_options_json: *const c_char,
        out_batch: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
}
