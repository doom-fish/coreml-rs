use core::ffi::c_void;
use std::ptr::NonNull;

use crate::ffi;

pub struct Retained(NonNull<c_void>);

unsafe impl Send for Retained {}

impl Retained {
    pub unsafe fn new(ptr: *mut c_void) -> Option<Self> {
        NonNull::new(ptr).map(Self)
    }

    pub fn into_raw(self) -> NonNull<c_void> {
        let ptr = self.0;
        core::mem::forget(self);
        ptr
    }
}

impl Drop for Retained {
    fn drop(&mut self) {
        unsafe { ffi::cm_object_release(self.0.as_ptr()) };
    }
}

#[cfg(feature = "async")]
pub struct CancelOnDrop(Option<Retained>);

#[cfg(feature = "async")]
impl CancelOnDrop {
    pub unsafe fn new(task: *mut c_void) -> Self {
        Self(unsafe { Retained::new(task) })
    }
}

#[cfg(feature = "async")]
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(task) = &self.0 {
            unsafe { ffi::cm_task_cancel(task.0.as_ptr()) };
        }
    }
}
