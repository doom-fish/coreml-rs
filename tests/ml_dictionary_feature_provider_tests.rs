use coreml::feature::Feature;
use coreml::ml_dictionary_feature_provider::MLDictionaryFeatureProvider;

#[test]
fn ml_dictionary_feature_provider_supports_generic_feature_insertion() {
    let mut provider = MLDictionaryFeatureProvider::new();
    let feature = Feature::from_string("doom fish").expect("string feature should build");
    provider.insert_feature("label", &feature);
    assert_eq!(provider.keys(), vec!["label".to_owned()]);
    assert_eq!(
        provider
            .get_feature("label")
            .and_then(|value| value.string_value()),
        Some("doom fish".to_owned())
    );
}
