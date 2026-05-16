//! `MLUpdateTask`-style synchronous update helpers.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::configuration::ModelConfiguration;
use crate::error::{from_swift, take_owned_c_string, CoreMLError};
use crate::feature_provider::BatchProvider;
use crate::ffi;

/// Progress events that may be requested from an update task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateEvent {
    /// Beginning of training.
    TrainingBegin,
    /// End of one epoch.
    EpochEnd,
    /// End of one mini-batch.
    MiniBatchEnd,
}

/// `MLTaskState` snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateTaskState {
    /// Task is suspended.
    #[default]
    Suspended,
    /// Task is running.
    Running,
    /// Task is cancelling.
    Cancelling,
    /// Task completed successfully.
    Completed,
    /// Task failed.
    Failed,
    /// Unknown state string from the bridge.
    Unknown,
}

/// Progress-handler configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateProgressHandlers {
    #[serde(default)]
    interested_events: Vec<UpdateEvent>,
}

impl UpdateProgressHandlers {
    /// Create an empty handler configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Request callbacks for one additional event.
    #[must_use]
    pub fn with_event(mut self, event: UpdateEvent) -> Self {
        if !self.interested_events.contains(&event) {
            self.interested_events.push(event);
        }
        self
    }

    /// Request callbacks for all public update events.
    #[must_use]
    pub fn all() -> Self {
        Self::new()
            .with_event(UpdateEvent::TrainingBegin)
            .with_event(UpdateEvent::EpochEnd)
            .with_event(UpdateEvent::MiniBatchEnd)
    }

    /// Ordered list of requested events.
    #[must_use]
    pub fn interested_events(&self) -> &[UpdateEvent] {
        &self.interested_events
    }

    fn as_json_c_string(&self) -> Result<CString, CoreMLError> {
        CString::new(serde_json::to_string(self).map_err(|error| {
            CoreMLError::UpdateFailed(format!(
                "failed to encode update progress handlers: {error}"
            ))
        })?)
        .map_err(|error| {
            CoreMLError::InvalidArgument(format!(
                "update progress handlers JSON contained an interior NUL byte: {error}"
            ))
        })
    }
}

/// Snapshot of one update callback context.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UpdateContext {
    /// Task identifier.
    #[serde(default)]
    pub task_identifier: Option<String>,
    /// Task state at callback time.
    #[serde(default)]
    pub state: UpdateTaskState,
    /// Progress/completion event name.
    pub event: String,
    /// Metrics reported by CoreML.
    #[serde(default)]
    pub metrics: BTreeMap<String, Value>,
    /// Parameters in effect at callback time.
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
    /// Error text, when the task failed.
    #[serde(default)]
    pub error_message: Option<String>,
}

/// Result of running an update task to completion.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UpdateResult {
    /// Final task state.
    #[serde(default)]
    pub final_state: UpdateTaskState,
    /// All captured progress/completion contexts.
    #[serde(default)]
    pub contexts: Vec<UpdateContext>,
}

/// Synchronous helpers for CoreML model update APIs.
#[derive(Debug, Default, Clone, Copy)]
pub struct Update;

impl Update {
    /// Run a CoreML update task to completion and return captured progress snapshots.
    ///
    /// # Errors
    ///
    /// Returns an error if task construction fails or CoreML reports an update failure.
    pub fn run(
        model_path: impl AsRef<Path>,
        training_data: &BatchProvider,
        configuration: Option<&ModelConfiguration>,
        handlers: &UpdateProgressHandlers,
    ) -> Result<UpdateResult, CoreMLError> {
        let model_path = path_to_c_string(model_path)?;
        let default_configuration = ModelConfiguration::default();
        let configuration_json = configuration
            .unwrap_or(&default_configuration)
            .as_json_c_string()?;
        let handlers_json = handlers.as_json_c_string()?;
        let mut error = ptr::null_mut();
        let mut result_json = ptr::null_mut();
        let status = unsafe {
            ffi::cm_update_run(
                model_path.as_ptr(),
                training_data.ptr,
                configuration_json.as_ptr(),
                handlers_json.as_ptr(),
                &mut result_json,
                &mut error,
            )
        };
        if status != ffi::status::OK || result_json.is_null() {
            return Err(from_swift(status, error));
        }
        serde_json::from_str(&take_owned_c_string(result_json)).map_err(|decode_error| {
            CoreMLError::UpdateFailed(format!(
                "failed to decode update result JSON: {decode_error}"
            ))
        })
    }
}

fn path_to_c_string(path: impl AsRef<Path>) -> Result<CString, CoreMLError> {
    CString::new(path.as_ref().to_string_lossy().into_owned()).map_err(|error| {
        CoreMLError::InvalidArgument(format!("path contains an interior NUL byte: {error}"))
    })
}
