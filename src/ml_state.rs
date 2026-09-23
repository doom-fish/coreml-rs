//! `MLState` handle wrapper.

use core::ffi::c_void;
use std::ffi::CString;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::ptr::{self, NonNull};
use std::thread;

use crate::error::{from_swift, CoreMLError};
use crate::ffi;
use crate::multi_array::{MultiArray, MultiArrayRef};

/// Handle to a CoreML state buffer set.
pub struct MLState {
    pub(crate) ptr: *mut c_void,
}

impl MLState {
    /// Whether the current runtime supports stateful CoreML inference APIs.
    #[must_use]
    pub fn runtime_supported() -> bool {
        unsafe { ffi::cm_state_runtime_supported() }
    }

    /// Copy a named state buffer into a standalone `MultiArray` snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime is too old or the state buffer is missing.
    pub fn snapshot_multi_array(&self, state_name: &str) -> Result<MultiArray, CoreMLError> {
        self.with_multi_array(state_name, MultiArrayRef::copy_to_owned)?
    }

    pub fn with_multi_array<R>(
        &self,
        state_name: &str,
        body: impl FnOnce(&MultiArrayRef) -> R,
    ) -> Result<R, CoreMLError> {
        visit_state_buffer(self.ptr, state_name, |buffer| {
            body(unsafe { MultiArrayRef::from_raw(buffer) })
        })
    }

    pub fn with_multi_array_mut<R>(
        &mut self,
        state_name: &str,
        body: impl FnOnce(&mut MultiArrayRef) -> R,
    ) -> Result<R, CoreMLError> {
        visit_state_buffer(self.ptr, state_name, |buffer| {
            body(unsafe { MultiArrayRef::from_raw_mut(buffer) })
        })
    }

    pub(crate) fn from_raw(ptr: *mut c_void) -> Option<Self> {
        (!ptr.is_null()).then_some(Self { ptr })
    }
}

struct StateBufferVisit<F, R> {
    body: Option<F>,
    outcome: Option<thread::Result<R>>,
}

unsafe extern "C" fn state_buffer_trampoline<F, R>(buffer: *mut c_void, context: *mut c_void)
where
    F: FnOnce(NonNull<c_void>) -> R,
{
    let Some(visit) = (unsafe { context.cast::<StateBufferVisit<F, R>>().as_mut() }) else {
        return;
    };
    let Some(buffer) = NonNull::new(buffer) else {
        return;
    };
    let Some(body) = visit.body.take() else {
        return;
    };
    visit.outcome = Some(catch_unwind(AssertUnwindSafe(|| body(buffer))));
}

fn visit_state_buffer<F, R>(state: *mut c_void, state_name: &str, body: F) -> Result<R, CoreMLError>
where
    F: FnOnce(NonNull<c_void>) -> R,
{
    let state_name_c = CString::new(state_name).map_err(|error| {
        CoreMLError::InvalidArgument(format!("state name contains an interior NUL byte: {error}"))
    })?;
    let mut visit = StateBufferVisit {
        body: Some(body),
        outcome: None,
    };
    let mut error = ptr::null_mut();
    let status = unsafe {
        ffi::cm_state_with_multi_array(
            state,
            state_name_c.as_ptr(),
            state_buffer_trampoline::<F, R>,
            (&raw mut visit).cast(),
            &raw mut error,
        )
    };
    match visit.outcome {
        Some(Ok(result)) => Ok(result),
        Some(Err(payload)) => resume_unwind(payload),
        None if status != ffi::status::OK => Err(from_swift(status, error)),
        None => Err(CoreMLError::StateFailed(format!(
            "state buffer '{state_name}' was unavailable"
        ))),
    }
}

impl Drop for MLState {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cm_object_release(self.ptr) };
        }
    }
}

impl core::fmt::Debug for MLState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MLState")
            .field("runtime_supported", &Self::runtime_supported())
            .finish()
    }
}
