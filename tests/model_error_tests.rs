use coreml::prelude::*;
use serde_json::Value;

#[test]
fn model_error_constants_match_sdk() {
    assert_eq!(ML_MODEL_ERROR_DOMAIN, "com.apple.CoreML");
    assert_eq!(
        MLModelError::from_code(11),
        Some(MLModelError::PredictionCancelled)
    );
    assert_eq!(MLModelError::CustomLayer.code(), 4);
}

#[test]
fn parameter_description_reconstructs_ml_key() {
    let description = ParameterDescription {
        key: "epochs".to_owned(),
        scope: Some("updater".to_owned()),
        default_value: Value::from(1),
        numeric_constraint: None,
    };

    let key = description.ml_key();
    assert_eq!(key.name, "epochs");
    assert_eq!(key.scope.as_deref(), Some("updater"));
}
