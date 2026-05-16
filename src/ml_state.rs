//! `MLState` handle wrapper.

use core::ffi::c_void;
use std::ffi::CString;
use std::ptr;

use crate::error::{from_swift, CoreMLError};
use crate::ffi;
use crate::multi_array::MultiArray;

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
        let state_name = CString::new(state_name).map_err(|error| {
            CoreMLError::InvalidArgument(format!(
                "state name contains an interior NUL byte: {error}"
            ))
        })?;
        let mut error = ptr::null_mut();
        let mut array = ptr::null_mut();
        let status = unsafe {
            ffi::cm_state_snapshot_multi_array(
                self.ptr,
                state_name.as_ptr(),
                &mut array,
                &mut error,
            )
        };
        if status != ffi::status::OK || array.is_null() {
            return Err(from_swift(status, error));
        }
        MultiArray::from_raw(array).ok_or_else(|| {
            CoreMLError::StateFailed("CoreML returned no multi-array state snapshot".to_owned())
        })
    }

    pub(crate) fn from_raw(ptr: *mut c_void) -> Option<Self> {
        (!ptr.is_null()).then_some(Self { ptr })
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
