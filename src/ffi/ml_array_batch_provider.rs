use core::ffi::c_void;

extern "C" {
    pub fn cm_batch_provider_new() -> *mut c_void;
    pub fn cm_batch_provider_push(batch: *mut c_void, provider: *mut c_void) -> i32;
}
