//! Feature-provider wrappers for model inputs, outputs, and batches.

use core::ffi::c_void;
use std::ffi::CString;

use apple_cf::cv::CVPixelBuffer;

use crate::error::{from_status_message, take_owned_c_string, CoreMLError};
use crate::feature::{Feature, FeatureType};
use crate::ffi;
use crate::multi_array::MultiArray;

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
    pub fn insert_feature(&mut self, name: &str, value: &Feature) {
        self.try_insert_feature(name, value)
            .expect("failed to insert feature into feature provider");
    }

    /// Insert an `MLMultiArray` value.
    pub fn insert_multi_array(&mut self, name: &str, value: MultiArray) {
        self.try_insert_multi_array(name, value)
            .expect("failed to insert MLMultiArray into feature provider");
    }

    /// Insert an image input from a `CVPixelBuffer`.
    pub fn insert_cv_pixel_buffer(&mut self, name: &str, value: &CVPixelBuffer) {
        self.try_insert_cv_pixel_buffer(name, value)
            .expect("failed to insert CVPixelBuffer into feature provider");
    }

    /// Insert a string feature.
    pub fn insert_string(&mut self, name: &str, value: &str) {
        self.try_insert_string(name, value)
            .expect("failed to insert string into feature provider");
    }

    /// Insert an `Int64` feature.
    pub fn insert_int64(&mut self, name: &str, value: i64) {
        self.try_insert_int64(name, value)
            .expect("failed to insert int64 into feature provider");
    }

    /// Insert a `Double` feature.
    pub fn insert_double(&mut self, name: &str, value: f64) {
        self.try_insert_double(name, value)
            .expect("failed to insert double into feature provider");
    }

    /// Fallible variant of [`insert_feature`](Self::insert_feature).
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name cannot cross the FFI boundary.
    pub fn try_insert_feature(&mut self, name: &str, value: &Feature) -> Result<(), CoreMLError> {
        let name = c_string(name, "feature name")?;
        let status = unsafe {
            ffi::cm_feature_provider_insert_feature(self.ptr, name.as_ptr(), value.as_ptr())
        };
        status_ok(status, "failed to insert feature")
    }

    /// Fallible variant of [`insert_multi_array`](Self::insert_multi_array).
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name cannot cross the FFI boundary.
    pub fn try_insert_multi_array(
        &mut self,
        name: &str,
        value: MultiArray,
    ) -> Result<(), CoreMLError> {
        let name = c_string(name, "feature name")?;
        let status = unsafe {
            ffi::cm_feature_provider_insert_multi_array(self.ptr, name.as_ptr(), value.as_ptr())
        };
        status_ok(status, "failed to insert multi-array")
    }

    /// Fallible variant of [`insert_cv_pixel_buffer`](Self::insert_cv_pixel_buffer).
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name cannot cross the FFI boundary.
    pub fn try_insert_cv_pixel_buffer(
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

    /// Fallible variant of [`insert_string`](Self::insert_string).
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name or value contains a NUL byte.
    pub fn try_insert_string(&mut self, name: &str, value: &str) -> Result<(), CoreMLError> {
        let name = c_string(name, "feature name")?;
        let value = c_string(value, "feature value")?;
        let status = unsafe {
            ffi::cm_feature_provider_insert_string(self.ptr, name.as_ptr(), value.as_ptr())
        };
        status_ok(status, "failed to insert string")
    }

    /// Fallible variant of [`insert_int64`](Self::insert_int64).
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name contains a NUL byte.
    pub fn try_insert_int64(&mut self, name: &str, value: i64) -> Result<(), CoreMLError> {
        let name = c_string(name, "feature name")?;
        let status =
            unsafe { ffi::cm_feature_provider_insert_int64(self.ptr, name.as_ptr(), value) };
        status_ok(status, "failed to insert int64")
    }

    /// Fallible variant of [`insert_double`](Self::insert_double).
    ///
    /// # Errors
    ///
    /// Returns an error when the feature name contains a NUL byte.
    pub fn try_insert_double(&mut self, name: &str, value: f64) -> Result<(), CoreMLError> {
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

    /// Fetch a retained `MLMultiArray` value.
    #[must_use]
    pub fn get_multi_array(&self, name: &str) -> Option<MultiArray> {
        let name = CString::new(name).ok()?;
        let ptr = unsafe { ffi::cm_feature_provider_get_multi_array(self.ptr, name.as_ptr()) };
        MultiArray::from_raw(ptr)
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
        unsafe { ffi::cm_feature_provider_get_int64(self.ptr, name.as_ptr(), &mut out) }
            .then_some(out)
    }

    /// Fetch a `Double` value.
    #[must_use]
    pub fn get_double(&self, name: &str) -> Option<f64> {
        let name = CString::new(name).ok()?;
        let mut out = 0.0_f64;
        unsafe { ffi::cm_feature_provider_get_double(self.ptr, name.as_ptr(), &mut out) }
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
        self.try_push(provider)
            .expect("failed to push feature provider into batch");
    }

    /// Fallible variant of [`push`](Self::push).
    ///
    /// # Errors
    ///
    /// Returns an error if the Swift bridge rejects the provider.
    pub fn try_push(&mut self, provider: FeatureProvider) -> Result<(), CoreMLError> {
        let status = unsafe { ffi::cm_batch_provider_push(self.ptr, provider.ptr) };
        status_ok(status, "failed to push feature provider into batch")
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
