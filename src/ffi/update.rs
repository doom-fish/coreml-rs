use core::ffi::{c_char, c_void};

pub type UpdateEventCallback = unsafe extern "C" fn(
    context: *mut c_void,
    kind: i32,
    context_json: *const c_char,
    model: *mut c_void,
);
pub type ContextReleaseCallback = unsafe extern "C" fn(context: *mut c_void);

pub const UPDATE_EVENT_PROGRESS: i32 = 0;
pub const UPDATE_EVENT_COMPLETION: i32 = 1;

extern "C" {
    pub fn cm_update_start(
        model_path: *const c_char,
        training_data: *mut c_void,
        configuration_json: *const c_char,
        handlers_json: *const c_char,
        callback: UpdateEventCallback,
        context: *mut c_void,
        release: ContextReleaseCallback,
        out_task: *mut *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
    pub fn cm_update_cancel(task: *mut c_void);
}
