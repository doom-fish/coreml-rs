use core::ffi::{c_char, c_void};

extern "C" {
    pub fn cm_custom_layer_register_class(
        class_name: *const c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_custom_layer_create(
        class_name: *const c_char,
        parameters_json: *const c_char,
        out_layer: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_custom_layer_set_weight_data(
        layer: *mut c_void,
        weight_data: *const *const u8,
        weight_lengths: *const usize,
        weight_count: usize,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_custom_layer_output_shapes_json(
        layer: *mut c_void,
        input_shapes_json: *const c_char,
        out_json: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_custom_layer_evaluate_cpu(
        layer: *mut c_void,
        inputs: *const *mut c_void,
        input_count: usize,
        outputs: *const *mut c_void,
        output_count: usize,
        error_out: *mut *mut c_char,
    ) -> i32;
}
