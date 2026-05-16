use coreml::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json = r#"{
        "inputs": [{
            "name": "image",
            "feature_type": "image",
            "optional": false,
            "image_constraint": {
                "pixels_wide": 224,
                "pixels_high": 224,
                "pixel_format_type": 1111970369,
                "size_constraint": {
                    "constraint_type": "enumerated",
                    "pixels_wide_range": {"lower": 224, "upper": 449},
                    "pixels_high_range": {"lower": 224, "upper": 449},
                    "enumerated_image_sizes": [
                        {"pixels_wide": 224, "pixels_high": 224},
                        {"pixels_wide": 448, "pixels_high": 448}
                    ]
                }
            }
        }, {
            "name": "tensor",
            "feature_type": "multi_array",
            "optional": false,
            "multi_array_constraint": {
                "shape": [1, 3],
                "data_type": "float32",
                "shape_constraint": {
                    "constraint_type": "range",
                    "size_ranges": [
                        {"lower": 1, "upper": 2},
                        {"lower": 3, "upper": 5}
                    ]
                }
            }
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
            "scope": "updater",
            "default_value": 1,
            "numeric_constraint": {"min": 1.0, "max": 8.0, "enumerated": [1.0, 2.0, 4.0]}
        }],
        "metadata": {"author": "doom fish"},
        "predicted_feature_name": "label",
        "predicted_probabilities_name": "scores",
        "class_labels": ["cat", "dog"],
        "is_updatable": true
    }"#;

    let description = DetailedModelDescription::from_json_str(json)?;
    assert_eq!(description.inputs.len(), 2);
    assert_eq!(
        description.parameter_descriptions[0]
            .ml_key()
            .scope
            .as_deref(),
        Some("updater")
    );
    println!(
        "parsed detailed description with flexible {:?} image sizing",
        description.inputs[0]
            .image_constraint
            .as_ref()
            .and_then(|constraint| constraint.size_constraint.as_ref())
            .map(|constraint| constraint.constraint_type)
    );
    Ok(())
}
