use coreml::prelude::*;

#[test]
fn detailed_model_description_decodes_flexible_constraints() {
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
                    ],
                    "enumerated_shapes": [[1, 3], [1, 4]]
                }
            }
        }],
        "outputs": [],
        "state_features": [],
        "training_inputs": [],
        "parameter_descriptions": [{
            "key": "epochs",
            "scope": "updater",
            "default_value": 1,
            "numeric_constraint": {"min": 1.0, "max": 10.0, "enumerated": [1.0, 2.0]}
        }],
        "metadata": {},
        "class_labels": [],
        "is_updatable": true
    }"#;

    let description = DetailedModelDescription::from_json_str(json)
        .expect("detailed description JSON should decode");

    let image_constraint = description.inputs[0]
        .image_constraint
        .as_ref()
        .expect("image constraint should be present");
    assert_eq!(
        image_constraint
            .size_constraint
            .as_ref()
            .unwrap()
            .constraint_type,
        ImageSizeConstraintType::Enumerated
    );
    assert_eq!(
        image_constraint
            .size_constraint
            .as_ref()
            .unwrap()
            .enumerated_image_sizes
            .len(),
        2
    );

    let shape_constraint = description.inputs[1]
        .multi_array_constraint
        .as_ref()
        .and_then(|constraint| constraint.shape_constraint.as_ref())
        .expect("shape constraint should be present");
    assert_eq!(
        shape_constraint.constraint_type,
        MultiArrayShapeConstraintType::Range
    );
    assert_eq!(shape_constraint.size_ranges[0].lower, 1);

    let key = description.parameter_descriptions[0].ml_key();
    assert_eq!(key.name, "epochs");
    assert_eq!(key.scope.as_deref(), Some("updater"));
}
