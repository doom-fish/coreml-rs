use coreml::feature_provider::FeatureProvider;
use coreml::ml_array_batch_provider::MLArrayBatchProvider;

fn main() {
    let mut first = FeatureProvider::new();
    first.insert_int64("value", 10);
    let mut second = FeatureProvider::new();
    second.insert_int64("value", 20);

    let batch = MLArrayBatchProvider::from_feature_providers(vec![first, second]);
    assert_eq!(batch.len(), 2);
    println!("array batch provider len = {}", batch.len());
}
