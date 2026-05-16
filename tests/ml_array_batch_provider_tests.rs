use coreml::feature_provider::FeatureProvider;
use coreml::ml_array_batch_provider::MLArrayBatchProvider;

#[test]
fn ml_array_batch_provider_builds_from_feature_providers() {
    let mut first = FeatureProvider::new();
    first.insert_int64("value", 10);
    let mut second = FeatureProvider::new();
    second.insert_int64("value", 20);

    let batch = MLArrayBatchProvider::from_feature_providers(vec![first, second]);
    assert_eq!(batch.len(), 2);
    assert_eq!(
        batch
            .get(1)
            .and_then(|provider| provider.get_int64("value")),
        Some(20)
    );
}
