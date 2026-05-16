use std::collections::BTreeMap;

use coreml::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let int_feature = Feature::from_int64(7)?;
    let string_feature = Feature::from_string("doom fish")?;
    assert_eq!(int_feature.int64_value(), Some(7));
    assert_eq!(string_feature.string_value().as_deref(), Some("doom fish"));

    let mut dictionary = BTreeMap::new();
    dictionary.insert("cat".to_owned(), 0.25);
    dictionary.insert("dog".to_owned(), 0.75);
    let dictionary_feature = Feature::from_string_dictionary(&dictionary)?;
    assert_eq!(
        dictionary_feature.string_dictionary_value(),
        Some(dictionary)
    );

    let mut array = MultiArray::new_f32(&[2])?;
    array.copy_from_f32_slice(&[1.0, 2.0])?;
    let array_feature = Feature::from_multi_array(array)?;
    assert_eq!(
        array_feature
            .multi_array_value()
            .unwrap()
            .as_f32_slice()
            .unwrap(),
        &[1.0, 2.0]
    );
    println!(
        "feature types: {:?}, {:?}",
        int_feature.feature_type(),
        array_feature.feature_type()
    );
    Ok(())
}
