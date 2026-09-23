use core::ffi::c_void;

extern "C" {
    pub fn cm_object_release(ptr: *mut c_void);
    pub fn cm_task_cancel(task: *mut c_void);
}
