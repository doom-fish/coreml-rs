use core::ffi::c_void;

extern "C" {
    pub fn cm_multi_array_new(
        shape: *const i64,
        rank: usize,
        data_type: i32,
        out_array: *mut *mut c_void,
        error_out: *mut *mut i8,
    ) -> i32;
    pub fn cm_multi_array_concat(
        arrays: *const *mut c_void,
        len: usize,
        axis: isize,
        data_type: i32,
        out_array: *mut *mut c_void,
        error_out: *mut *mut i8,
    ) -> i32;
    pub fn cm_multi_array_transfer_to(
        source: *mut c_void,
        destination: *mut c_void,
        error_out: *mut *mut i8,
    ) -> i32;
    pub fn cm_multi_array_count(array: *mut c_void) -> usize;
    pub fn cm_multi_array_data_type(array: *mut c_void) -> i32;
    pub fn cm_multi_array_rank(array: *mut c_void) -> usize;
    pub fn cm_multi_array_copy_shape(
        array: *mut c_void,
        out_buffer: *mut i64,
        capacity: usize,
    ) -> usize;
    pub fn cm_multi_array_copy_strides(
        array: *mut c_void,
        out_buffer: *mut i64,
        capacity: usize,
    ) -> usize;
    pub fn cm_multi_array_data_pointer(array: *mut c_void) -> *mut c_void;
}
