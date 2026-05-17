use std::collections::BTreeMap;

use coreml::prelude::*;
use serde_json::{json, Value};

struct ScaleModel {
    scale: f64,
    saw_empty_description: bool,
}

impl MLCustomModel for ScaleModel {
    fn prediction_from_features(
        &mut self,
        input: &FeatureProvider,
        options: &PredictionOptions,
    ) -> Result<FeatureProvider, CoreMLError> {
        let value = input.get_double("value").ok_or_else(|| {
            CoreMLError::CustomModelFailed("expected an input feature named 'value'".to_owned())
        })?;
        let mut output = FeatureProvider::new();
        output
            .try_insert_double(
                "value",
                value.mul_add(self.scale, if options.uses_cpu_only() { 1.0 } else { 0.0 }),
            )
            .map_err(|error| {
                CoreMLError::CustomModelFailed(format!(
                    "failed to populate custom-model output feature provider: {error}"
                ))
            })?;
        output.try_insert_string(
            "description_state",
            if self.saw_empty_description {
                "empty"
            } else {
                "non_empty"
            },
        )?;
        Ok(output)
    }
}

fn model_parameters(scale: f64) -> BTreeMap<String, Value> {
    BTreeMap::from([(String::from("scale"), json!(scale))])
}

fn input_provider(value: f64) -> FeatureProvider {
    let mut provider = FeatureProvider::new();
    provider.insert_double("value", value);
    provider
}

#[test]
#[ignore = "run via examples/17_ml_custom_model.rs; direct integration-test worker threads crash CoreML custom callbacks"]
fn custom_model_registration_round_trips_single_and_batch_predictions() {
    let registration = MLCustomModelRegistration::register("RustTestScaleModel", |context| {
        let scale = context
            .parameters
            .get("scale")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        Ok(ScaleModel {
            scale,
            saw_empty_description: context.model_description.inputs.is_empty()
                && context.model_description.outputs.is_empty(),
        })
    })
    .expect("custom model should register");

    let mut model = registration
        .instantiate(&model_parameters(3.0))
        .expect("custom model should instantiate");

    let output = model
        .predict(
            &input_provider(2.0),
            &PredictionOptions::new().with_uses_cpu_only(true),
        )
        .expect("single custom-model prediction should succeed");
    assert_eq!(output.get_double("value"), Some(7.0));
    assert_eq!(
        output.get_string("description_state"),
        Some(String::from("empty"))
    );

    let batch =
        BatchProvider::from_feature_providers(vec![input_provider(1.0), input_provider(-2.0)]);
    let batch_output = model
        .predict_batch(&batch, &PredictionOptions::default())
        .expect("batch custom-model prediction should succeed");

    assert_eq!(batch_output.len(), 2);
    assert_eq!(
        batch_output
            .get(0)
            .and_then(|provider| provider.get_double("value")),
        Some(3.0)
    );
    assert_eq!(
        batch_output
            .get(1)
            .and_then(|provider| provider.get_double("value")),
        Some(-6.0)
    );
}

#[test]
fn custom_model_registration_rejects_duplicate_class_names() {
    let _registration = MLCustomModelRegistration::register("RustDuplicateModel", |_context| {
        Ok(ScaleModel {
            scale: 1.0,
            saw_empty_description: true,
        })
    })
    .expect("first custom model registration should succeed");

    let error = MLCustomModelRegistration::register("RustDuplicateModel", |_context| {
        Ok(ScaleModel {
            scale: 1.0,
            saw_empty_description: true,
        })
    })
    .expect_err("duplicate custom model registrations must fail");

    assert!(matches!(error, CoreMLError::InvalidArgument(_)));
}
