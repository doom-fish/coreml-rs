use coreml::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
                            "inputs": {"x": {"bindings": [{"name": "input"}]}},
                            "outputs": [{"name": "output", "value_type": {"description": "tensor<float32>"}}],
                            "blocks": []
                        }]
                    }
                }
            }
        }
    }"#;

    let structure = ModelStructure::from_json_str(json)?;
    println!("model structure kind: {:?}", structure.kind);
    Ok(())
}
