#![cfg(feature = "async")]

mod common;

use std::error::Error;

use coreml::prelude::*;

#[test]
fn async_predict_compiled_model_succeeds() -> Result<(), Box<dyn Error>> {
    let source = common::asset_path("sentiment_classifier.mlmodel");
    let compiled = common::compile_model(&source, "async-predict");

    let model = pollster::block_on(Model::load_async(compiled.as_path(), None))?;
    let mut inputs = FeatureProvider::new();
    inputs.insert_string("text", "I love this product");

    let outputs = pollster::block_on(model.predict_async(&inputs, None))?;
    assert_eq!(outputs.get_string("label").as_deref(), Some("positive"));
    Ok(())
}
