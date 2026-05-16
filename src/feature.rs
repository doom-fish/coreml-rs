//! `MLFeatureValue` wrapper.

use core::ffi::c_void;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{from_status_message, from_swift, take_owned_c_string, CoreMLError};
use crate::ffi;
use crate::ml_sequence::MLSequence;
use crate::multi_array::MultiArray;

/// Public CoreML feature-value kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureType {
    /// Invalid or unknown feature type.
    Invalid,
    /// Signed 64-bit integer feature.
    Int64,
    /// Double-precision floating-point feature.
    Double,
    /// UTF-8 string feature.
    String,
    /// Image / `CVPixelBuffer` feature.
    Image,
    /// `MLMultiArray` feature.
    MultiArray,
    /// Dictionary-valued feature.
    Dictionary,
    /// Sequence feature.
    Sequence,
    /// Stateful model feature.
    State,
}

impl FeatureType {
    pub(crate) fn from_ffi(raw: i32) -> Option<Self> {
        match raw {
            0 => Some(Self::Invalid),
            1 => Some(Self::Int64),
            2 => Some(Self::Double),
            3 => Some(Self::String),
            4 => Some(Self::Image),
            5 => Some(Self::MultiArray),
            6 => Some(Self::Dictionary),
            7 => Some(Self::Sequence),
            8 => Some(Self::State),
            _ => None,
        }
    }

    pub(crate) const fn as_ffi(self) -> i32 {
        match self {
            Self::Invalid => 0,
            Self::Int64 => 1,
            Self::Double => 2,
            Self::String => 3,
            Self::Image => 4,
            Self::MultiArray => 5,
            Self::Dictionary => 6,
            Self::Sequence => 7,
            Self::State => 8,
        }
    }
}

/// Normalized crop rectangle used with image feature conversion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImageCropRect {
    /// Normalized x origin.
    pub x: f64,
    /// Normalized y origin.
    pub y: f64,
    /// Normalized width.
    pub width: f64,
    /// Normalized height.
    pub height: f64,
}

/// Crop-and-scale strategies accepted by CoreML image conversion helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageCropAndScale {
    /// Crop around the center after preserving aspect ratio.
    CenterCrop,
    /// Fit the image while preserving aspect ratio.
    ScaleFit,
    /// Fill the destination size.
    ScaleFill,
    /// Scale-to-fit after a 90° counter-clockwise rotation.
    ScaleFitRotate90Ccw,
    /// Scale-to-fill after a 90° counter-clockwise rotation.
    ScaleFillRotate90Ccw,
}

/// Optional knobs for `MLFeatureValue` image conversion.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageFeatureOptions {
    /// Optional normalized crop rect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop_rect: Option<ImageCropRect>,
    /// Optional crop-and-scale strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop_and_scale: Option<ImageCropAndScale>,
}

impl ImageFeatureOptions {
    fn as_json_c_string(&self) -> Result<CString, CoreMLError> {
        CString::new(serde_json::to_string(self).map_err(|error| {
            CoreMLError::InvalidArgument(format!("failed to encode image feature options: {error}"))
        })?)
        .map_err(|error| {
            CoreMLError::InvalidArgument(format!(
                "image feature options JSON contained an interior NUL byte: {error}"
            ))
        })
    }
}

/// Owned wrapper around `MLFeatureValue`.
pub struct Feature {
    ptr: *mut c_void,
}

impl Feature {
    /// Create an `Int64` feature value.
    pub fn from_int64(value: i64) -> Result<Self, CoreMLError> {
        Self::from_direct_ptr(
            unsafe { ffi::cm_feature_new_int64(value) },
            "failed to create Int64 feature",
        )
    }

    /// Create a `Double` feature value.
    pub fn from_double(value: f64) -> Result<Self, CoreMLError> {
        Self::from_direct_ptr(
            unsafe { ffi::cm_feature_new_double(value) },
            "failed to create Double feature",
        )
    }

    /// Create a `String` feature value.
    pub fn from_string(value: &str) -> Result<Self, CoreMLError> {
        let value = CString::new(value).map_err(|error| {
            CoreMLError::InvalidArgument(format!(
                "feature string contains an interior NUL byte: {error}"
            ))
        })?;
        Self::from_direct_ptr(
            unsafe { ffi::cm_feature_new_string(value.as_ptr()) },
            "failed to create String feature",
        )
    }

    /// Create a `MultiArray` feature value.
    pub fn from_multi_array(value: MultiArray) -> Result<Self, CoreMLError> {
        Self::from_direct_ptr(
            unsafe { ffi::cm_feature_new_multi_array(value.as_ptr()) },
            "failed to create MultiArray feature",
        )
    }

    /// Create an image feature value from an image on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is invalid or CoreML cannot convert the image.
    pub fn from_image_url(
        path: impl AsRef<Path>,
        pixels_wide: usize,
        pixels_high: usize,
        pixel_format_type: u32,
    ) -> Result<Self, CoreMLError> {
        Self::from_image_url_with_options(
            path,
            pixels_wide,
            pixels_high,
            pixel_format_type,
            &ImageFeatureOptions::default(),
        )
    }

    /// Create an image feature value from an image on disk with explicit conversion options.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is invalid or CoreML cannot convert the image.
    pub fn from_image_url_with_options(
        path: impl AsRef<Path>,
        pixels_wide: usize,
        pixels_high: usize,
        pixel_format_type: u32,
        options: &ImageFeatureOptions,
    ) -> Result<Self, CoreMLError> {
        let path = path_to_c_string(path)?;
        let options_json = options.as_json_c_string()?;
        let mut error = ptr::null_mut();
        let ptr = unsafe {
            ffi::cm_feature_new_image_at_url(
                path.as_ptr(),
                pixels_wide,
                pixels_high,
                pixel_format_type,
                options_json.as_ptr(),
                &mut error,
            )
        };
        if ptr.is_null() {
            return Err(from_swift(ffi::status::FEATURE_PROVIDER_FAILED, error));
        }
        Ok(Self { ptr })
    }

    /// Create a sequence feature value.
    pub fn from_sequence(value: MLSequence) -> Result<Self, CoreMLError> {
        Self::from_direct_ptr(
            unsafe { ffi::cm_feature_new_sequence(value.as_ptr()) },
            "failed to create Sequence feature",
        )
    }

    /// Create an undefined feature value of the requested type.
    pub fn undefined(feature_type: FeatureType) -> Result<Self, CoreMLError> {
        Self::from_direct_ptr(
            unsafe { ffi::cm_feature_new_undefined(feature_type.as_ffi()) },
            "failed to create undefined feature",
        )
    }

    /// Create a dictionary feature with `String` keys.
    pub fn from_string_dictionary(entries: &BTreeMap<String, f64>) -> Result<Self, CoreMLError> {
        let keys: Vec<CString> = entries
            .keys()
            .map(|key| CString::new(key.as_str()))
            .collect::<Result<_, _>>()
            .map_err(|error| {
                CoreMLError::InvalidArgument(format!(
                    "dictionary key contains an interior NUL byte: {error}"
                ))
            })?;
        let key_ptrs: Vec<*const i8> = keys.iter().map(|key| key.as_ptr()).collect();
        let values: Vec<f64> = entries.values().copied().collect();
        let mut error = ptr::null_mut();
        let ptr = unsafe {
            ffi::cm_feature_new_string_dictionary(
                key_ptrs.as_ptr(),
                values.as_ptr(),
                values.len(),
                &mut error,
            )
        };
        if ptr.is_null() {
            return Err(from_swift(ffi::status::FEATURE_PROVIDER_FAILED, error));
        }
        Ok(Self { ptr })
    }

    /// Create a dictionary feature with `Int64` keys.
    pub fn from_int64_dictionary(entries: &BTreeMap<i64, f64>) -> Result<Self, CoreMLError> {
        let keys: Vec<i64> = entries.keys().copied().collect();
        let values: Vec<f64> = entries.values().copied().collect();
        let mut error = ptr::null_mut();
        let ptr = unsafe {
            ffi::cm_feature_new_int64_dictionary(
                keys.as_ptr(),
                values.as_ptr(),
                values.len(),
                &mut error,
            )
        };
        if ptr.is_null() {
            return Err(from_swift(ffi::status::FEATURE_PROVIDER_FAILED, error));
        }
        Ok(Self { ptr })
    }

    /// Feature type.
    #[must_use]
    pub fn feature_type(&self) -> FeatureType {
        FeatureType::from_ffi(unsafe { ffi::cm_feature_type(self.ptr) })
            .unwrap_or(FeatureType::Invalid)
    }

    /// Whether this feature represents an undefined value.
    #[must_use]
    pub fn is_undefined(&self) -> bool {
        unsafe { ffi::cm_feature_is_undefined(self.ptr) }
    }

    /// Retrieve the `Int64` value when present.
    #[must_use]
    pub fn int64_value(&self) -> Option<i64> {
        let mut out = 0_i64;
        unsafe { ffi::cm_feature_get_int64(self.ptr, &mut out) }.then_some(out)
    }

    /// Retrieve the `Double` value when present.
    #[must_use]
    pub fn double_value(&self) -> Option<f64> {
        let mut out = 0.0_f64;
        unsafe { ffi::cm_feature_get_double(self.ptr, &mut out) }.then_some(out)
    }

    /// Retrieve the `String` value when present.
    #[must_use]
    pub fn string_value(&self) -> Option<String> {
        let ptr = unsafe { ffi::cm_feature_get_string(self.ptr) };
        (!ptr.is_null()).then(|| take_owned_c_string(ptr))
    }

    /// Retrieve a retained `MultiArray` value when present.
    #[must_use]
    pub fn multi_array_value(&self) -> Option<MultiArray> {
        let ptr = unsafe { ffi::cm_feature_get_multi_array(self.ptr) };
        MultiArray::from_raw(ptr)
    }

    /// Retrieve a retained `MLSequence` value when present.
    #[must_use]
    pub fn sequence_value(&self) -> Option<MLSequence> {
        let ptr = unsafe { ffi::cm_feature_get_sequence(self.ptr) };
        MLSequence::from_raw(ptr)
    }

    /// Retrieve a string-keyed dictionary value when present.
    #[must_use]
    pub fn string_dictionary_value(&self) -> Option<BTreeMap<String, f64>> {
        let ptr = unsafe { ffi::cm_feature_get_string_dictionary_json(self.ptr) };
        (!ptr.is_null())
            .then(|| serde_json::from_str(&take_owned_c_string(ptr)).ok())
            .flatten()
    }

    /// Retrieve an int-keyed dictionary value when present.
    #[must_use]
    pub fn int64_dictionary_value(&self) -> Option<BTreeMap<i64, f64>> {
        let ptr = unsafe { ffi::cm_feature_get_int64_dictionary_json(self.ptr) };
        let json = (!ptr.is_null()).then(|| take_owned_c_string(ptr))?;
        let map: BTreeMap<String, f64> = serde_json::from_str(&json).ok()?;
        let mut parsed = BTreeMap::new();
        for (key, value) in map {
            parsed.insert(key.parse().ok()?, value);
        }
        Some(parsed)
    }

    /// Check CoreML equality semantics for two feature values.
    #[must_use]
    pub fn is_equal(&self, other: &Self) -> bool {
        unsafe { ffi::cm_feature_is_equal(self.ptr, other.ptr) }
    }

    pub(crate) const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub(crate) fn from_raw(ptr: *mut c_void) -> Option<Self> {
        (!ptr.is_null()).then_some(Self { ptr })
    }

    fn from_direct_ptr(ptr: *mut c_void, fallback_message: &str) -> Result<Self, CoreMLError> {
        if ptr.is_null() {
            Err(from_status_message(
                ffi::status::FEATURE_PROVIDER_FAILED,
                fallback_message.to_owned(),
            ))
        } else {
            Ok(Self { ptr })
        }
    }
}

impl Drop for Feature {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cm_object_release(self.ptr) };
        }
    }
}

impl core::fmt::Debug for Feature {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_struct("Feature");
        debug
            .field("feature_type", &self.feature_type())
            .field("undefined", &self.is_undefined());
        match self.feature_type() {
            FeatureType::Int64 => {
                debug.field("value", &self.int64_value());
            }
            FeatureType::Double => {
                debug.field("value", &self.double_value());
            }
            FeatureType::String => {
                debug.field("value", &self.string_value());
            }
            FeatureType::Dictionary => {
                let string_value = self.string_dictionary_value();
                let int_value = self.int64_dictionary_value();
                if string_value.is_some() {
                    debug.field("value", &string_value);
                } else {
                    debug.field("value", &int_value);
                }
            }
            FeatureType::MultiArray => {
                debug.field("value", &self.multi_array_value());
            }
            FeatureType::Sequence => {
                debug.field("value", &self.sequence_value());
            }
            _ => {}
        }
        debug.finish()
    }
}

impl TryFrom<&Feature> for Value {
    type Error = CoreMLError;

    fn try_from(feature: &Feature) -> Result<Self, Self::Error> {
        match feature.feature_type() {
            FeatureType::Int64 => Ok(feature.int64_value().map_or(Value::Null, Value::from)),
            FeatureType::Double => Ok(feature.double_value().map_or(Value::Null, Value::from)),
            FeatureType::String => Ok(feature.string_value().map_or(Value::Null, Value::from)),
            FeatureType::Dictionary => {
                if let Some(dict) = feature.string_dictionary_value() {
                    Ok(serde_json::to_value(dict).map_err(|error| {
                        CoreMLError::FeatureProviderFailed(format!(
                            "failed to encode string dictionary feature: {error}"
                        ))
                    })?)
                } else if let Some(dict) = feature.int64_dictionary_value() {
                    Ok(serde_json::to_value(dict).map_err(|error| {
                        CoreMLError::FeatureProviderFailed(format!(
                            "failed to encode int64 dictionary feature: {error}"
                        ))
                    })?)
                } else {
                    Ok(Value::Null)
                }
            }
            FeatureType::Sequence => {
                if let Some(sequence) = feature.sequence_value() {
                    if let Some(strings) = sequence.string_values() {
                        Ok(serde_json::to_value(strings).map_err(|error| {
                            CoreMLError::FeatureProviderFailed(format!(
                                "failed to encode string sequence feature: {error}"
                            ))
                        })?)
                    } else if let Some(ints) = sequence.int64_values() {
                        Ok(serde_json::to_value(ints).map_err(|error| {
                            CoreMLError::FeatureProviderFailed(format!(
                                "failed to encode int64 sequence feature: {error}"
                            ))
                        })?)
                    } else {
                        Ok(Value::Null)
                    }
                } else {
                    Ok(Value::Null)
                }
            }
            _ => Err(CoreMLError::FeatureProviderFailed(format!(
                "feature type {:?} is not convertible to serde_json::Value",
                feature.feature_type()
            ))),
        }
    }
}

fn path_to_c_string(path: impl AsRef<Path>) -> Result<CString, CoreMLError> {
    CString::new(path.as_ref().to_string_lossy().into_owned()).map_err(|error| {
        CoreMLError::InvalidArgument(format!("path contains an interior NUL byte: {error}"))
    })
}
