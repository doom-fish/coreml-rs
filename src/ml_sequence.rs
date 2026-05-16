//! `MLSequence` wrapper.

use core::ffi::c_void;
use std::ffi::CString;
use std::ptr;

use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::feature::FeatureType;
use crate::ffi;

/// Owned wrapper around `MLSequence`.
pub struct MLSequence {
    pub(crate) ptr: *mut c_void,
}

impl MLSequence {
    /// Create an empty sequence for the requested element type.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested type is not supported by `MLSequence`.
    pub fn empty(feature_type: FeatureType) -> Result<Self, CoreMLError> {
        validate_feature_type(feature_type)?;
        let mut error = ptr::null_mut();
        let ptr = unsafe { ffi::cm_sequence_new_empty(feature_type.as_ffi(), &mut error) };
        if ptr.is_null() {
            return Err(from_swift(ffi::status::FEATURE_PROVIDER_FAILED, error));
        }
        Ok(Self { ptr })
    }

    /// Create a string-valued sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if any string contains an interior NUL byte.
    pub fn from_strings<S: AsRef<str>>(values: &[S]) -> Result<Self, CoreMLError> {
        let strings: Vec<CString> = values
            .iter()
            .map(|value| CString::new(value.as_ref()))
            .collect::<Result<_, _>>()
            .map_err(|error| {
                CoreMLError::InvalidArgument(format!(
                    "sequence string contains an interior NUL byte: {error}"
                ))
            })?;
        let pointers: Vec<*const i8> = strings.iter().map(|value| value.as_ptr()).collect();
        let mut error = ptr::null_mut();
        let ptr =
            unsafe { ffi::cm_sequence_new_strings(pointers.as_ptr(), pointers.len(), &mut error) };
        if ptr.is_null() {
            return Err(from_swift(ffi::status::FEATURE_PROVIDER_FAILED, error));
        }
        Ok(Self { ptr })
    }

    /// Create an `Int64` sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML rejects the values.
    pub fn from_int64s(values: &[i64]) -> Result<Self, CoreMLError> {
        let mut error = ptr::null_mut();
        let ptr = unsafe { ffi::cm_sequence_new_int64s(values.as_ptr(), values.len(), &mut error) };
        if ptr.is_null() {
            return Err(from_swift(ffi::status::FEATURE_PROVIDER_FAILED, error));
        }
        Ok(Self { ptr })
    }

    /// Element type carried by the sequence.
    #[must_use]
    pub fn feature_type(&self) -> FeatureType {
        FeatureType::from_ffi(unsafe { ffi::cm_sequence_type(self.ptr) })
            .unwrap_or(FeatureType::Invalid)
    }

    /// Retrieve all string elements when the sequence is string-backed.
    #[must_use]
    pub fn string_values(&self) -> Option<Vec<String>> {
        let ptr = unsafe { ffi::cm_sequence_get_strings_json(self.ptr) };
        let json = (!ptr.is_null()).then(|| take_owned_c_string(ptr))?;
        serde_json::from_str(&json).ok()
    }

    /// Retrieve all `Int64` elements when the sequence is int-backed.
    #[must_use]
    pub fn int64_values(&self) -> Option<Vec<i64>> {
        let ptr = unsafe { ffi::cm_sequence_get_int64s_json(self.ptr) };
        let json = (!ptr.is_null()).then(|| take_owned_c_string(ptr))?;
        serde_json::from_str(&json).ok()
    }

    pub(crate) const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub(crate) fn from_raw(ptr: *mut c_void) -> Option<Self> {
        (!ptr.is_null()).then_some(Self { ptr })
    }
}

impl Drop for MLSequence {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cm_object_release(self.ptr) };
        }
    }
}

impl core::fmt::Debug for MLSequence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_struct("MLSequence");
        debug.field("feature_type", &self.feature_type());
        match self.feature_type() {
            FeatureType::String => {
                debug.field("values", &self.string_values());
            }
            FeatureType::Int64 => {
                debug.field("values", &self.int64_values());
            }
            _ => {}
        }
        debug.finish()
    }
}

fn validate_feature_type(feature_type: FeatureType) -> Result<(), CoreMLError> {
    match feature_type {
        FeatureType::String | FeatureType::Int64 => Ok(()),
        other => Err(CoreMLError::InvalidArgument(format!(
            "MLSequence supports only string and int64 values, got {other:?}"
        ))),
    }
}
