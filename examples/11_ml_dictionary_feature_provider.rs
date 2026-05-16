use coreml::feature::Feature;
use coreml::ml_dictionary_feature_provider::MLDictionaryFeatureProvider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut provider = MLDictionaryFeatureProvider::new();
    let feature = Feature::from_string("doom fish")?;
    provider.insert_feature("label", &feature);
    assert_eq!(
        provider
            .get_feature("label")
            .and_then(|value| value.string_value()),
        Some("doom fish".to_owned())
    );
    println!("dictionary provider keys: {:?}", provider.keys());
    Ok(())
}
