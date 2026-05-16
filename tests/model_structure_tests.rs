use coreml::prelude::*;

#[test]
fn model_structure_decodes_program_snapshot() {
    let json = r#"{
        "kind": "program",
        "program": {
            "functions": {
                "main": {
                    "inputs": [{"name": "input", "value_type": {"description": "tensor<float32>"}}],
                    "block": {
                        "inputs": [{"name": "input", "value_type": {"description": "tensor<float32>"}}],
                        "output_names": ["output"],
                        "operations": [{
                            "operator_name": "relu",
                            "inputs": {
                                "x": {"bindings": [{"name": "input"}]}
                            },
                            "outputs": [{"name": "output", "value_type": {"description": "tensor<float32>"}}],
                            "blocks": []
                        }]
                    }
                }
            }
        }
    }"#;

    let structure =
        ModelStructure::from_json_str(json).expect("model-structure JSON should decode");
    assert_eq!(structure.kind, ModelStructureKind::Program);
    let program = structure
        .program
        .expect("program structure should be present");
    let function = program
        .functions
        .get("main")
        .expect("main function should exist");
    assert_eq!(function.block.output_names, ["output"]);
    assert_eq!(function.block.operations[0].operator_name, "relu");
}

#[test]
fn compute_plan_details_decode_costs_and_devices() {
    let json = r#"{
        "model_type": "program",
        "function_names": ["main"],
        "operation_count": 1,
        "layer_count": 0,
        "pipeline_model_count": 0,
        "model_structure": {
            "kind": "program",
            "program": {
                "functions": {
                    "main": {
                        "inputs": [{"name": "input", "value_type": {"description": "tensor<float32>"}}],
                        "block": {
                            "inputs": [],
                            "output_names": ["output"],
                            "operations": [{
                                "operator_name": "relu",
                                "inputs": {"x": {"bindings": [{"name": "input"}]}},
                                "outputs": [{"name": "output", "value_type": {"description": "tensor<float32>"}}],
                                "blocks": []
                            }]
                        }
                    }
                }
            }
        },
        "program_operation_plans": [{
            "path": "functions.main.block.operations[0]",
            "operation": {
                "operator_name": "relu",
                "inputs": {"x": {"bindings": [{"name": "input"}]}},
                "outputs": [{"name": "output", "value_type": {"description": "tensor<float32>"}}],
                "blocks": []
            },
            "cost": {"weight": 0.5},
            "device_usage": {
                "supported": [{"kind": "cpu", "description": "CPU"}],
                "preferred": {"kind": "cpu", "description": "CPU"}
            }
        }],
        "neural_network_layer_plans": []
    }"#;

    let details =
        ComputePlanDetails::from_json_str(json).expect("detailed compute-plan JSON should decode");
    assert_eq!(details.summary.model_type, ComputePlanModelType::Program);
    assert_eq!(details.summary.operation_count, 1);
    assert!(
        (details.program_operation_plans[0].cost.unwrap().weight - 0.5).abs() < f64::EPSILON,
        "cost weight should round-trip"
    );
    assert_eq!(
        details.program_operation_plans[0]
            .device_usage
            .as_ref()
            .expect("device usage should be present")
            .preferred
            .kind,
        ComputeDeviceKind::Cpu
    );
}

#[test]
fn model_structure_missing_bundle_fails() {
    let error = ModelStructure::load_from_url("tests/does-not-exist.mlmodelc")
        .expect_err("missing model-structure bundle should fail");
    assert!(matches!(
        error,
        CoreMLError::DescriptionFailed(_)
            | CoreMLError::Unsupported(_)
            | CoreMLError::Unknown { .. }
    ));
}
