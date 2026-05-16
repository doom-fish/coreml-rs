//! `MLPredictionOptions` builder.

use std::ffi::CString;

use serde::{Deserialize, Serialize};

use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::ffi;

/// Safe Rust builder for `MLPredictionOptions`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionOptions {
    #[serde(default)]
    uses_cpu_only: bool,
}

/// Snapshot of the bridge-normalized prediction options.
pub type PredictionOptionsBridgeSnapshot = PredictionOptions;

impl PredictionOptions {
    /// Create prediction options with CoreML defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Force prediction onto CPU only.
    #[must_use]
    pub const fn with_uses_cpu_only(mut self, uses_cpu_only: bool) -> Self {
        self.uses_cpu_only = uses_cpu_only;
        self
    }

    /// Whether the deprecated `usesCPUOnly` flag is enabled.
    #[must_use]
    pub const fn uses_cpu_only(&self) -> bool {
        self.uses_cpu_only
    }

    /// Round-trip these options through the Swift bridge.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization or bridge decoding fails.
    pub fn bridge_snapshot(&self) -> Result<PredictionOptionsBridgeSnapshot, CoreMLError> {
        let json = self.as_json_c_string()?;
        let mut error = std::ptr::null_mut();
        let snapshot =
            unsafe { ffi::cm_prediction_options_snapshot_json(json.as_ptr(), &mut error) };
        if snapshot.is_null() {
            return Err(from_swift(ffi::status::PREDICTION_FAILED, error));
        }
        serde_json::from_str(&take_owned_c_string(snapshot)).map_err(|decode_error| {
            CoreMLError::PredictionFailed(format!(
                "failed to decode prediction-options snapshot JSON: {decode_error}"
            ))
        })
    }

    pub(crate) fn as_json_c_string(&self) -> Result<CString, CoreMLError> {
        CString::new(serde_json::to_string(self).map_err(|error| {
            CoreMLError::PredictionFailed(format!("failed to encode prediction options: {error}"))
        })?)
        .map_err(|error| {
            CoreMLError::InvalidArgument(format!(
                "prediction options JSON contained an interior NUL byte: {error}"
            ))
        })
    }
}
