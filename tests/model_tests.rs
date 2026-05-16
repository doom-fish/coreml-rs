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
