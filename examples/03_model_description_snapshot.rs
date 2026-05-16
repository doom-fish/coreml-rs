use coreml::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json = r#"{
        "inputs": [{
            "name": "image",
            "feature_type": "multi_array",
            "optional": false,
            "multi_array_constraint": {"shape": [1, 3], "data_type": "float32"}
        }],
        "outputs": [{
            "name": "scores",
            "feature_type": "dictionary",
            "optional": false,
            "dictionary_constraint": {"key_type": "string"}
        }],
        "state_features": [{
            "name": "accumulator",
            "feature_type": "state",
            "optional": false,
            "state_constraint": {"buffer_shape": [1], "data_type": "float32"}
        }],
        "training_inputs": [{"name": "label", "feature_type": "string", "optional": false}],
        "parameter_descriptions": [{
            "key": "epochs",
            "default_value": 1,
            "numeric_constraint": {"min": 1.0, "max": 8.0, "enumerated": [1.0, 2.0, 4.0]}
        }],
        "metadata": {"author": "doom fish"},
        "predicted_feature_name": "label",
        "predicted_probabilities_name": "scores",
        "class_labels": ["cat", "dog"],
        "is_updatable": true
    }"#;

    let description = ModelDescription::from_json_str(json)?;
    assert_eq!(description.inputs.len(), 1);
    assert_eq!(description.outputs.len(), 1);
    assert_eq!(description.state_features.len(), 1);
    assert!(description.is_updatable);
    println!(
        "parsed model description with {} input(s)",
        description.inputs.len()
    );
    Ok(())
}
