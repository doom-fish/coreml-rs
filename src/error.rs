//! Errors produced by the CoreML bridge.

use core::ffi::c_char;
use core::fmt;

use libc::free;

use crate::ffi;

/// Top-level error type returned by this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoreMLError {
    /// Invalid input crossed the FFI boundary.
    InvalidArgument(String),
    /// Loading a compiled model failed.
    ModelLoadFailed(String),
    /// Running inference failed.
    PredictionFailed(String),
    /// Compiling a `.mlmodel` into `.mlmodelc` failed.
    CompilationFailed(String),
    /// Building or querying a feature provider failed.
    FeatureProviderFailed(String),
    /// Constructing or mutating an `MLMultiArray` failed.
    MultiArrayFailed(String),
    /// Creating or loading an `MLModelAsset` from in-memory bytes failed.
    ModelAssetFailed(String),
    /// The requested operation is unsupported on this system.
    Unsupported(String),
    /// An async CoreML bridge operation timed out.
    TimedOut(String),
    /// The requested indices are outside the array bounds.
    IndexOutOfRange(String),
    /// A value had the wrong CoreML type for the requested operation.
    TypeMismatch(String),
    /// Catch-all for unmapped Swift / Objective-C errors.
    Unknown { code: i32, message: String },
}

impl CoreMLError {
    /// Numeric status code reported by the Swift bridge.
    #[must_use]
    pub const fn code(&self) -> i32 {
        match self {
            Self::InvalidArgument(_) => ffi::status::INVALID_ARGUMENT,
            Self::ModelLoadFailed(_) => ffi::status::MODEL_LOAD_FAILED,
            Self::PredictionFailed(_) => ffi::status::PREDICTION_FAILED,
            Self::CompilationFailed(_) => ffi::status::COMPILATION_FAILED,
            Self::FeatureProviderFailed(_) => ffi::status::FEATURE_PROVIDER_FAILED,
            Self::MultiArrayFailed(_) | Self::TypeMismatch(_) => ffi::status::MULTI_ARRAY_FAILED,
            Self::ModelAssetFailed(_) => ffi::status::MODEL_ASSET_FAILED,
            Self::Unsupported(_) => ffi::status::UNSUPPORTED,
            Self::TimedOut(_) => ffi::status::TIMED_OUT,
            Self::IndexOutOfRange(_) => ffi::status::INDEX_OUT_OF_RANGE,
            Self::Unknown { code, .. } => *code,
        }
    }

    /// Human-readable description from Swift / Objective-C.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::InvalidArgument(message)
            | Self::ModelLoadFailed(message)
            | Self::PredictionFailed(message)
            | Self::CompilationFailed(message)
            | Self::FeatureProviderFailed(message)
            | Self::MultiArrayFailed(message)
            | Self::ModelAssetFailed(message)
            | Self::Unsupported(message)
            | Self::TimedOut(message)
            | Self::IndexOutOfRange(message)
            | Self::TypeMismatch(message)
            | Self::Unknown { message, .. } => message,
        }
    }
}

impl fmt::Display for CoreMLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (code {})", self.message(), self.code())
    }
}

impl std::error::Error for CoreMLError {}

/// Take ownership of a Swift-allocated C string and free it with `libc::free`.
pub(crate) fn take_owned_c_string(ptr: *mut c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }

    let string = unsafe { core::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { free(ptr.cast()) };
    string
}

/// Build a `CoreMLError` from a Swift status code + optional error message.
pub(crate) fn from_swift(status: i32, error_str: *mut c_char) -> CoreMLError {
    let message = take_owned_c_string(error_str);
    from_status_message(status, message)
}

/// Build a `CoreMLError` from a status code and message generated in Rust.
pub(crate) fn from_status_message(status: i32, message: String) -> CoreMLError {
    match status {
        ffi::status::INVALID_ARGUMENT => CoreMLError::InvalidArgument(message),
        ffi::status::MODEL_LOAD_FAILED => CoreMLError::ModelLoadFailed(message),
        ffi::status::PREDICTION_FAILED => CoreMLError::PredictionFailed(message),
        ffi::status::COMPILATION_FAILED => CoreMLError::CompilationFailed(message),
        ffi::status::FEATURE_PROVIDER_FAILED => CoreMLError::FeatureProviderFailed(message),
        ffi::status::MULTI_ARRAY_FAILED => CoreMLError::MultiArrayFailed(message),
        ffi::status::MODEL_ASSET_FAILED => CoreMLError::ModelAssetFailed(message),
        ffi::status::UNSUPPORTED => CoreMLError::Unsupported(message),
        ffi::status::TIMED_OUT => CoreMLError::TimedOut(message),
        ffi::status::INDEX_OUT_OF_RANGE => CoreMLError::IndexOutOfRange(message),
        code => CoreMLError::Unknown { code, message },
    }
}
