mod common;

use coreml::prelude::*;

#[test]
fn model_load_missing_bundle_fails() {
    let configuration = ModelConfiguration::new();
    let error = Model::load_from_url("tests/does-not-exist.mlmodelc", &configuration)
        .expect_err("missing model bundle should fail");
    assert!(matches!(
        error,
        CoreMLError::ModelLoadFailed(_) | CoreMLError::Unknown { .. }
    ));
}

#[test]
fn model_rejects_empty_specification() {
    let error = Model::load_from_specification_data(&[], &ModelConfiguration::new())
        .expect_err("empty specification should fail");
    assert!(matches!(error, CoreMLError::InvalidArgument(_)));
}

#[test]
fn non_utf8_paths_are_rejected_instead_of_rewritten() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    let path = Path::new(OsStr::from_bytes(b"tests/not-\xFF-utf8.mlmodelc"));
    let error = Model::load_from_url(path, &ModelConfiguration::new())
        .expect_err("a non-UTF-8 path must not be converted lossily");
    assert!(matches!(error, CoreMLError::InvalidArgument(_)), "{error}");
    let error = ModelCompiler::compile(path).expect_err("non-UTF-8 source path");
    assert!(matches!(error, CoreMLError::InvalidArgument(_)), "{error}");
}

#[test]
fn model_descriptions_decode_or_report_errors() {
    let compiled = common::compile_model(
        &common::asset_path("sentiment_classifier.mlmodel"),
        "model-description",
    );
    let model = Model::load_from_url(&compiled, &ModelConfiguration::new()).unwrap();
    let description = model.description().expect("description should decode");
    assert_eq!(description.inputs[0].name, "text");
    let detailed = model
        .detailed_description()
        .expect("detailed description should decode");
    assert_eq!(detailed.outputs[0].name, "label");
    assert!(matches!(
        ModelDescription::from_json_str("{\"inputs\": 5}"),
        Err(CoreMLError::DescriptionFailed(_))
    ));
}
