//! Compute-device discovery snapshots.

use core::ffi::c_char;
use std::ptr;

use serde::{Deserialize, Serialize};

use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::ffi;

/// Broad CoreML compute-device classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeDeviceKind {
    /// CPU execution.
    Cpu,
    /// GPU execution.
    Gpu,
    /// Apple Neural Engine execution.
    NeuralEngine,
    /// Future / unknown device type surfaced by the framework.
    Unknown,
}

/// Snapshot of one CoreML compute device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeDevice {
    /// Device class.
    pub kind: ComputeDeviceKind,
    /// Framework-provided human-readable description.
    pub description: String,
    /// Core count for Apple Neural Engine devices, when known.
    #[serde(default)]
    pub total_core_count: Option<usize>,
}

/// Discover all CoreML-visible compute devices on the current machine.
///
/// # Errors
///
/// Returns an error when the runtime does not support compute-device discovery.
pub fn all_compute_devices() -> Result<Vec<ComputeDevice>, CoreMLError> {
    let mut error = ptr::null_mut();
    let mut json = ptr::null_mut();
    let status = unsafe { ffi::cm_all_compute_devices_json(&mut json, &mut error) };
    decode_device_list(status, json, error)
}

pub(crate) fn decode_device_list(
    status: i32,
    json: *mut c_char,
    error: *mut c_char,
) -> Result<Vec<ComputeDevice>, CoreMLError> {
    if status != ffi::status::OK || json.is_null() {
        return Err(from_swift(status, error));
    }
    let json = take_owned_c_string(json);
    serde_json::from_str(&json).map_err(|decode_error| {
        CoreMLError::DescriptionFailed(format!(
            "failed to decode compute-device JSON: {decode_error}"
        ))
    })
}
