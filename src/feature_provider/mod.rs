//! Feature-provider wrappers for model inputs, outputs, and batches.

use core::ffi::c_void;
use std::ffi::CString;

use apple_cf::cv::CVPixelBuffer;

use crate::error::{from_status_message, take_owned_c_string, CoreMLError};
use crate::feature::{Feature, FeatureType};
use crate::ffi;
use crate::multi_array::{MultiArray, MultiArrayView};

/// Mutable dictionary-style model input / output bag.
pub struct FeatureProvider {
    pub(crate) ptr: *mut c_void,
}

impl FeatureProvider {
    /// Create an empty feature provider.
    #[must_use]
    pub fn new() -> Self {
        let ptr = unsafe { ffi::cm_feature_provider_new() };
        assert!(!ptr.is_null(), "CoreML feature-provider allocation failed");
        Self { ptr }
    }

    /// Insert a generic feature value.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name cannot cross the FFI boundary.
    pub fn insert_feature(&mut self, name: &str, value: &Feature) -> Result<(), CoreMLError> {
        let name = c_string(name, "feature name")?;
        let status = unsafe {
            ffi::cm_feature_provider_insert_feature(self.ptr, name.as_ptr(), value.as_ptr())
        };
        status_ok(status, "failed to insert feature")
    }

    /// Insert an `MLMultiArray` value.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name cannot cross the FFI boundary.
    pub fn insert_multi_array(&mut self, name: &str, value: MultiArray) -> Result<(), CoreMLError> {
        let name = c_string(name, "feature name")?;
        let status = unsafe {
            ffi::cm_feature_provider_insert_multi_array(self.ptr, name.as_ptr(), value.as_ptr())
        };
        status_ok(status, "failed to insert multi-array")
    }

    /// Insert an image input from a `CVPixelBuffer`.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name cannot cross the FFI boundary.
    pub fn insert_cv_pixel_buffer(
        &mut self,
        name: &str,
        value: &CVPixelBuffer,
    ) -> Result<(), CoreMLError> {
        let name = c_string(name, "feature name")?;
        let status = unsafe {
            ffi::cm_feature_provider_insert_pixel_buffer(
                self.ptr,
                name.as_ptr(),
                value.as_ptr().cast::<c_void>(),
            )
        };
        status_ok(status, "failed to insert pixel buffer")
    }

    /// Insert a string feature.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name or value contains a NUL byte.
    pub fn insert_string(&mut self, name: &str, value: &str) -> Result<(), CoreMLError> {
        let name = c_string(name, "feature name")?;
        let value = c_string(value, "feature value")?;
        let status = unsafe {
            ffi::cm_feature_provider_insert_string(self.ptr, name.as_ptr(), value.as_ptr())
        };
        status_ok(status, "failed to insert string")
    }

    /// Insert an `Int64` feature.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name contains a NUL byte.
    pub fn insert_int64(&mut self, name: &str, value: i64) -> Result<(), CoreMLError> {
        let name = c_string(name, "feature name")?;
        let status =
            unsafe { ffi::cm_feature_provider_insert_int64(self.ptr, name.as_ptr(), value) };
        status_ok(status, "failed to insert int64")
    }

    /// Insert a `Double` feature.
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name contains a NUL byte.
    pub fn insert_double(&mut self, name: &str, value: f64) -> Result<(), CoreMLError> {
        let name = c_string(name, "feature name")?;
        let status =
            unsafe { ffi::cm_feature_provider_insert_double(self.ptr, name.as_ptr(), value) };
        status_ok(status, "failed to insert double")
    }

    /// Lookup the feature type for a named value.
    #[must_use]
    pub fn feature_type(&self, name: &str) -> Option<FeatureType> {
        let name = CString::new(name).ok()?;
        let raw = unsafe { ffi::cm_feature_provider_feature_type(self.ptr, name.as_ptr()) };
        FeatureType::from_ffi(raw)
    }

    /// Fetch a retained `Feature` value.
    #[must_use]
    pub fn get_feature(&self, name: &str) -> Option<Feature> {
        let name = CString::new(name).ok()?;
        let ptr = unsafe { ffi::cm_feature_provider_get_feature(self.ptr, name.as_ptr()) };
        Feature::from_raw(ptr)
    }

    /// Fetch a read-only view of an `MLMultiArray` value.
    #[must_use]
    pub fn get_multi_array(&self, name: &str) -> Option<MultiArrayView<'_>> {
        let name = CString::new(name).ok()?;
        let ptr = unsafe { ffi::cm_feature_provider_get_multi_array(self.ptr, name.as_ptr()) };
        unsafe { MultiArrayView::from_retained(ptr) }
    }

    /// Fetch a string value.
    #[must_use]
    pub fn get_string(&self, name: &str) -> Option<String> {
        let name = CString::new(name).ok()?;
        let ptr = unsafe { ffi::cm_feature_provider_get_string(self.ptr, name.as_ptr()) };
        (!ptr.is_null()).then(|| take_owned_c_string(ptr))
    }

    /// Fetch an `Int64` value.
    #[must_use]
    pub fn get_int64(&self, name: &str) -> Option<i64> {
        let name = CString::new(name).ok()?;
        let mut out = 0_i64;
        unsafe { ffi::cm_feature_provider_get_int64(self.ptr, name.as_ptr(), &raw mut out) }
            .then_some(out)
    }

    /// Fetch a `Double` value.
    #[must_use]
    pub fn get_double(&self, name: &str) -> Option<f64> {
        let name = CString::new(name).ok()?;
        let mut out = 0.0_f64;
        unsafe { ffi::cm_feature_provider_get_double(self.ptr, name.as_ptr(), &raw mut out) }
            .then_some(out)
    }

    /// Sorted feature names currently present in the provider.
    #[must_use]
    pub fn keys(&self) -> Vec<String> {
        let json = unsafe { ffi::cm_feature_provider_keys_json(self.ptr) };
        serde_json::from_str(&take_owned_c_string(json)).unwrap_or_default()
    }

    pub(crate) fn from_raw(ptr: *mut c_void) -> Option<Self> {
        (!ptr.is_null()).then_some(Self { ptr })
    }
}

impl Default for FeatureProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FeatureProvider {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cm_object_release(self.ptr) };
        }
    }
}

impl core::fmt::Debug for FeatureProvider {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FeatureProvider")
            .field("keys", &self.keys())
            .finish()
    }
}

/// Owned wrapper around `MLArrayBatchProvider` / `MLBatchProvider`.
pub struct BatchProvider {
    pub(crate) ptr: *mut c_void,
}

impl BatchProvider {
    /// Create an empty batch.
    #[must_use]
    pub fn new() -> Self {
        let ptr = unsafe { ffi::cm_batch_provider_new() };
        assert!(!ptr.is_null(), "CoreML batch-provider allocation failed");
        Self { ptr }
    }

    /// Create a batch from an iterator of feature providers.
    #[must_use]
    pub fn from_feature_providers(providers: impl IntoIterator<Item = FeatureProvider>) -> Self {
        let mut batch = Self::new();
        for provider in providers {
            batch.push(provider);
        }
        batch
    }

    /// Append one feature provider to the batch.
    pub fn push(&mut self, provider: FeatureProvider) {
        let status = unsafe { ffi::cm_batch_provider_push(self.ptr, provider.ptr) };
        debug_assert_eq!(status, ffi::status::OK);
    }

    /// Number of feature providers in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { ffi::cm_batch_provider_count(self.ptr) }
    }

    /// Whether the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fetch a feature provider by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<FeatureProvider> {
        let ptr = unsafe { ffi::cm_batch_provider_get_provider(self.ptr, index) };
        FeatureProvider::from_raw(ptr)
    }

    pub(crate) fn from_raw(ptr: *mut c_void) -> Option<Self> {
        (!ptr.is_null()).then_some(Self { ptr })
    }
}

impl Default for BatchProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BatchProvider {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cm_object_release(self.ptr) };
        }
    }
}

impl core::fmt::Debug for BatchProvider {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BatchProvider")
            .field("len", &self.len())
            .finish()
    }
}

fn c_string(value: &str, field_name: &str) -> Result<CString, CoreMLError> {
    CString::new(value).map_err(|error| {
        CoreMLError::InvalidArgument(format!(
            "{field_name} contains an interior NUL byte: {error}"
        ))
    })
}

fn status_ok(status: i32, fallback_message: &str) -> Result<(), CoreMLError> {
    if status == ffi::status::OK {
        Ok(())
    } else {
        Err(from_status_message(status, fallback_message.to_owned()))
    }
}
