use coreml::feature::Feature;
use coreml::ml_dictionary_feature_provider::MLDictionaryFeatureProvider;

#[test]
fn ml_dictionary_feature_provider_supports_generic_feature_insertion() {
    let mut provider = MLDictionaryFeatureProvider::new();
    let feature = Feature::from_string("doom fish").expect("string feature should build");
    provider.insert_feature("label", &feature).unwrap();
    assert_eq!(provider.keys(), vec!["label".to_owned()]);
    assert_eq!(
        provider
            .get_feature("label")
            .and_then(|value| value.string_value()),
        Some("doom fish".to_owned())
    );
}

#[test]
fn inserting_names_or_values_with_nul_bytes_is_an_error_not_a_panic() {
    use coreml::error::CoreMLError;
    use coreml::multi_array::MultiArray;

    let mut provider = MLDictionaryFeatureProvider::new();
    let feature = Feature::from_int64(1).unwrap();
    assert!(matches!(
        provider.insert_feature("bad\0name", &feature),
        Err(CoreMLError::InvalidArgument(_))
    ));
    assert!(matches!(
        provider.insert_string("name", "bad\0value"),
        Err(CoreMLError::InvalidArgument(_))
    ));
    assert!(matches!(
        provider.insert_int64("bad\0name", 1),
        Err(CoreMLError::InvalidArgument(_))
    ));
    assert!(matches!(
        provider.insert_double("bad\0name", 1.0),
        Err(CoreMLError::InvalidArgument(_))
    ));
    assert!(matches!(
        provider.insert_multi_array("bad\0name", MultiArray::new_f32(&[1]).unwrap()),
        Err(CoreMLError::InvalidArgument(_))
    ));
    assert!(provider.keys().is_empty());
}
