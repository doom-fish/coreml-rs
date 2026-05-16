use coreml::prelude::*;

#[test]
fn model_description_decodes_extended_snapshot() {
    let json = r#"{
        "inputs": [{
            "name": "input",
            "feature_type": "multi_array",
            "optional": false,
            "multi_array_constraint": {"shape": [1, 4], "data_type": "float32"}
        }],
        "outputs": [{
            "name": "output",
            "feature_type": "dictionary",
            "optional": false,
            "dictionary_constraint": {"key_type": "string"}
        }],
        "state_features": [{
            "name": "state",
            "feature_type": "state",
            "optional": false,
            "state_constraint": {"buffer_shape": [1], "data_type": "float32"}
        }],
        "training_inputs": [{"name": "label", "feature_type": "string", "optional": false}],
        "parameter_descriptions": [{
            "key": "epochs",
            "default_value": 1,
            "numeric_constraint": {"min": 1.0, "max": 10.0, "enumerated": [1.0, 2.0]}
        }],
        "metadata": {"author": "doom fish"},
        "predicted_feature_name": "label",
        "predicted_probabilities_name": "output",
        "class_labels": ["cat", "dog"],
        "is_updatable": true
    }"#;

    let description =
        ModelDescription::from_json_str(json).expect("description JSON should decode");
    assert_eq!(description.inputs[0].name, "input");
    assert_eq!(description.outputs[0].feature_type, FeatureType::Dictionary);
    assert_eq!(description.state_features.len(), 1);
    assert_eq!(description.parameter_descriptions[0].key, "epochs");
    assert!(description.is_updatable);
}
