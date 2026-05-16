use core::ffi::c_void;

extern "C" {
    pub fn cm_batch_provider_count(batch: *mut c_void) -> usize;
    pub fn cm_batch_provider_get_provider(batch: *mut c_void, index: usize) -> *mut c_void;
}
