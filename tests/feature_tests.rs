use std::collections::BTreeMap;

use coreml::prelude::*;

#[test]
fn feature_round_trips_scalars_and_dictionaries() {
    let int_feature = Feature::from_int64(7).expect("int64 feature should build");
    let double_feature = Feature::from_double(0.25).expect("double feature should build");
    let string_feature = Feature::from_string("doom fish").expect("string feature should build");

    assert_eq!(int_feature.int64_value(), Some(7));
    assert_eq!(double_feature.double_value(), Some(0.25));
    assert_eq!(string_feature.string_value().as_deref(), Some("doom fish"));

    let mut string_dictionary = BTreeMap::new();
    string_dictionary.insert("cat".to_owned(), 0.4);
    string_dictionary.insert("dog".to_owned(), 0.6);
    let dictionary_feature = Feature::from_string_dictionary(&string_dictionary)
        .expect("dictionary feature should build");
    assert_eq!(
        dictionary_feature.string_dictionary_value(),
        Some(string_dictionary)
    );

    let mut int_dictionary = BTreeMap::new();
    int_dictionary.insert(1_i64, 0.3);
    int_dictionary.insert(2_i64, 0.7);
    let int_dictionary_feature = Feature::from_int64_dictionary(&int_dictionary)
        .expect("int-keyed dictionary feature should build");
    assert_eq!(
        int_dictionary_feature.int64_dictionary_value(),
        Some(int_dictionary)
    );
}

#[test]
fn feature_wraps_multi_array_values() {
    let mut array = MultiArray::new_f32(&[2]).expect("multi-array should allocate");
    array
        .copy_from_f32_slice(&[1.0, 2.0])
        .expect("multi-array should accept data");
    let feature = Feature::from_multi_array(array).expect("feature should wrap multi-array");
    assert_eq!(feature.feature_type(), FeatureType::MultiArray);
    assert_eq!(
        feature.multi_array_value().unwrap().as_f32_slice().unwrap(),
        &[1.0, 2.0]
    );
}
