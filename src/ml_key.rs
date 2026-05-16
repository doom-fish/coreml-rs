//! Public wrapper for CoreML `MLKey` values.

use serde::{Deserialize, Serialize};

use crate::model_description::ParameterDescription;

/// Snapshot of an `MLKey` (`name` + optional `scope`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MLKey {
    /// Key name.
    pub name: String,
    /// Optional scoped suffix.
    #[serde(default)]
    pub scope: Option<String>,
}

impl MLKey {
    /// Create a new `MLKey` snapshot.
    #[must_use]
    pub fn new(name: impl Into<String>, scope: Option<String>) -> Self {
        Self {
            name: name.into(),
            scope,
        }
    }
}

impl From<&ParameterDescription> for MLKey {
    fn from(value: &ParameterDescription) -> Self {
        Self::new(value.key.clone(), value.scope.clone())
    }
}
