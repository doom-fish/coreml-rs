use core::ffi::{c_char, c_void};

pub type ModelAsyncCallback =
    extern "C" fn(status: i32, result: *mut c_void, error: *const c_char, user_data: *mut c_void);

extern "C" {
    pub fn cm_model_load(
        path: *const c_char,
        configuration_json: *const c_char,
        out_model: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_load_async(
        path: *const c_char,
        configuration_json: *const c_char,
        callback: ModelAsyncCallback,
        user_data: *mut c_void,
    );
    pub fn cm_model_load_from_specification(
        bytes: *const u8,
        byte_count: usize,
        configuration_json: *const c_char,
        out_model: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_available_compute_devices_json(
        out_json: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_write_to_url(
        model: *mut c_void,
        path: *const c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_predict(
        model: *mut c_void,
        inputs: *mut c_void,
        out_provider: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_predict_async(
        model: *mut c_void,
        inputs: *mut c_void,
        prediction_options_json: *const c_char,
        callback: ModelAsyncCallback,
        user_data: *mut c_void,
    );
    pub fn cm_model_predict_with_options(
        model: *mut c_void,
        inputs: *mut c_void,
        prediction_options_json: *const c_char,
        out_provider: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_predict_batch(
        model: *mut c_void,
        batch: *mut c_void,
        out_batch: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_predict_batch_with_options(
        model: *mut c_void,
        batch: *mut c_void,
        prediction_options_json: *const c_char,
        out_batch: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_new_state(
        model: *mut c_void,
        out_state: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_model_predict_with_state(
        model: *mut c_void,
        inputs: *mut c_void,
        state: *mut c_void,
        prediction_options_json: *const c_char,
        out_provider: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
}
