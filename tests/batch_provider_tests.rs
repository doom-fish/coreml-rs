use coreml::prelude::*;

#[test]
fn batch_provider_collects_feature_providers() {
    let mut first = FeatureProvider::new();
    first.insert_int64("count", 1);
    let mut second = FeatureProvider::new();
    second.insert_string("label", "doom fish");

    let batch = BatchProvider::from_feature_providers(vec![first, second]);
    assert_eq!(batch.len(), 2);
    assert_eq!(
        batch
            .get(0)
            .and_then(|provider| provider.get_int64("count")),
        Some(1)
    );
    assert_eq!(
        batch
            .get(1)
            .and_then(|provider| provider.get_string("label")),
        Some("doom fish".to_owned())
    );
}
