#![cfg(feature = "async")]

mod common;

use std::error::Error;

use coreml::prelude::*;

#[test]
fn async_load_compiled_model_succeeds() -> Result<(), Box<dyn Error>> {
    let source = common::asset_path("sentiment_classifier.mlmodel");
    let compiled = common::compile_model(&source, "async-load");
    let configuration = ModelConfiguration::new();

    let model = pollster::block_on(Model::load_async(compiled.as_path(), Some(&configuration)))?;
    let description = model.description();

    assert_eq!(description.inputs.len(), 1);
    assert_eq!(description.inputs[0].name, "text");
    assert_eq!(description.outputs.len(), 1);
    assert_eq!(description.outputs[0].name, "label");
    Ok(())
}
