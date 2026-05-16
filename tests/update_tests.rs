use coreml::prelude::*;

#[test]
fn update_missing_bundle_fails() {
    let mut provider = FeatureProvider::new();
    provider.insert_double("score", 0.5);
    let batch = BatchProvider::from_feature_providers(vec![provider]);
    let handlers = UpdateProgressHandlers::all();
    let error = Update::run("tests/does-not-exist.mlmodelc", &batch, None, &handlers)
        .expect_err("missing update model should fail");
    assert!(matches!(
        error,
        CoreMLError::UpdateFailed(_) | CoreMLError::Unknown { .. }
    ));
}
