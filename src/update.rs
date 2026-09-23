//! `MLUpdateTask`-style synchronous update helpers.

use core::ffi::{c_char, c_void};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use doom_fish_utils::callback_context::CallbackContext;
use doom_fish_utils::panic_safe::catch_user_panic;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::configuration::ModelConfiguration;
use crate::error::{from_swift, CoreMLError};
use crate::feature_provider::BatchProvider;
use crate::ffi;
use crate::model::Model;
use crate::retained::Retained;

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

type ProgressHandler<'a> = Box<dyn FnMut(&UpdateContext) + 'a>;

/// Progress-handler configuration.
#[derive(Default)]
pub struct UpdateProgressHandlers<'a> {
    interested_events: Vec<UpdateEvent>,
    on_progress: Option<ProgressHandler<'a>>,
}

impl<'a> UpdateProgressHandlers<'a> {
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

    #[must_use]
    pub fn on_progress(mut self, handler: impl FnMut(&UpdateContext) + 'a) -> Self {
        self.on_progress = Some(Box::new(handler));
        self
    }

    fn as_json_c_string(&self) -> Result<CString, CoreMLError> {
        #[derive(Serialize)]
        struct Payload<'e> {
            interested_events: &'e [UpdateEvent],
        }

        let payload = Payload {
            interested_events: &self.interested_events,
        };
        CString::new(serde_json::to_string(&payload).map_err(|error| {
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

impl core::fmt::Debug for UpdateProgressHandlers<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UpdateProgressHandlers")
            .field("interested_events", &self.interested_events)
            .field("on_progress", &self.on_progress.is_some())
            .finish()
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

#[derive(Debug)]
pub struct UpdateOutcome {
    pub model: Model,
    pub result: UpdateResult,
}

/// Synchronous helpers for CoreML model update APIs.
#[derive(Debug, Default, Clone, Copy)]
pub struct Update;

impl Update {
    /// Run a CoreML update task to completion and return the updated model and captured progress
    /// snapshots.
    ///
    /// # Errors
    ///
    /// Returns an error if task construction fails, CoreML reports an update failure, or the
    /// timeout elapses.
    pub fn run(
        model_path: impl AsRef<Path>,
        training_data: &BatchProvider,
        configuration: Option<&ModelConfiguration>,
        mut handlers: UpdateProgressHandlers<'_>,
        timeout: Option<Duration>,
    ) -> Result<UpdateOutcome, CoreMLError> {
        let model_path = path_to_c_string(model_path)?;
        let default_configuration = ModelConfiguration::default();
        let configuration_json = configuration
            .unwrap_or(&default_configuration)
            .as_json_c_string()?;
        let handlers_json = handlers.as_json_c_string()?;
        let deadline = timeout.and_then(|timeout| Instant::now().checked_add(timeout));
        let (sender, receiver) = mpsc::channel();
        let events = CallbackContext::new(sender);
        let mut error = ptr::null_mut();
        let mut task = ptr::null_mut();
        let status = unsafe {
            ffi::cm_update_start(
                model_path.as_ptr(),
                training_data.ptr,
                configuration_json.as_ptr(),
                handlers_json.as_ptr(),
                update_event_trampoline,
                events.retained_ptr(),
                CallbackContext::<UpdateSender>::RELEASE,
                &raw mut task,
                &raw mut error,
            )
        };
        if status != ffi::status::OK {
            if !task.is_null() {
                unsafe { ffi::cm_object_release(task) };
            }
            return Err(from_swift(status, error));
        }
        let _task = NonNull::new(task).map(UpdateTask).ok_or_else(|| {
            CoreMLError::UpdateFailed("CoreML returned no update task".to_owned())
        })?;

        let mut contexts = Vec::new();
        loop {
            let message = match deadline {
                Some(deadline) => {
                    match receiver.recv_timeout(deadline.saturating_duration_since(Instant::now()))
                    {
                        Ok(message) => message,
                        Err(RecvTimeoutError::Timeout) => {
                            return Err(CoreMLError::TimedOut(
                                "CoreML update task did not finish before the timeout and was cancelled"
                                    .to_owned(),
                            ));
                        }
                        Err(RecvTimeoutError::Disconnected) => return Err(update_disconnected()),
                    }
                }
                None => receiver.recv().map_err(|_| update_disconnected())?,
            };
            match message {
                UpdateMessage::Progress(json) => {
                    let context = decode_context(&json)?;
                    if let Some(on_progress) = handlers.on_progress.as_mut() {
                        on_progress(&context);
                    }
                    contexts.push(context);
                }
                UpdateMessage::Completion { json, model } => {
                    let context = decode_context(&json)?;
                    let final_state = context.state;
                    let failure = context.error_message.clone();
                    contexts.push(context);
                    return match (final_state, model) {
                        (UpdateTaskState::Completed, Some(model)) => Ok(UpdateOutcome {
                            model: unsafe { Model::from_retained(model.into_raw()) },
                            result: UpdateResult {
                                final_state,
                                contexts,
                            },
                        }),
                        (UpdateTaskState::Failed, _) => Err(CoreMLError::UpdateFailed(
                            failure.unwrap_or_else(|| "CoreML update task failed".to_owned()),
                        )),
                        (state, _) => Err(CoreMLError::UpdateFailed(format!(
                            "CoreML update task finished in state {state:?} without an updated model"
                        ))),
                    };
                }
            }
        }
    }
}

enum UpdateMessage {
    Progress(String),
    Completion {
        json: String,
        model: Option<Retained>,
    },
}

type UpdateSender = mpsc::Sender<UpdateMessage>;

struct UpdateTask(NonNull<c_void>);

impl Drop for UpdateTask {
    fn drop(&mut self) {
        unsafe {
            ffi::cm_update_cancel(self.0.as_ptr());
            ffi::cm_object_release(self.0.as_ptr());
        }
    }
}

unsafe extern "C" fn update_event_trampoline(
    context: *mut c_void,
    kind: i32,
    context_json: *const c_char,
    model: *mut c_void,
) {
    let model = unsafe { Retained::new(model) };
    catch_user_panic("coreml::update_event", move || {
        let json = if context_json.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(context_json) }
                .to_string_lossy()
                .into_owned()
        };
        let message = if kind == ffi::UPDATE_EVENT_COMPLETION {
            UpdateMessage::Completion { json, model }
        } else {
            drop(model);
            UpdateMessage::Progress(json)
        };
        unsafe {
            CallbackContext::<UpdateSender>::with(context, "coreml::update_event", move |sender| {
                let _ = sender.send(message);
            });
        }
    });
}

fn decode_context(json: &str) -> Result<UpdateContext, CoreMLError> {
    serde_json::from_str(json).map_err(|error| {
        CoreMLError::UpdateFailed(format!("failed to decode update context JSON: {error}"))
    })
}

fn update_disconnected() -> CoreMLError {
    CoreMLError::UpdateFailed("CoreML update task stopped reporting progress".to_owned())
}

fn path_to_c_string(path: impl AsRef<Path>) -> Result<CString, CoreMLError> {
    let path = path.as_ref();
    let path = path.to_str().ok_or_else(|| {
        CoreMLError::InvalidArgument(format!("path is not valid UTF-8: {}", path.display()))
    })?;
    CString::new(path).map_err(|error| {
        CoreMLError::InvalidArgument(format!("path contains an interior NUL byte: {error}"))
    })
}
