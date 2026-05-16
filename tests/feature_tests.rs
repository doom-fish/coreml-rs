use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use coreml::prelude::*;

const ONE_BY_ONE_PNG: [u8; 68] = [
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 4, 0,
    0, 0, 181, 28, 12, 2, 0, 0, 0, 11, 73, 68, 65, 84, 120, 218, 99, 252, 255, 31, 0, 3, 3, 2, 0,
    239, 191, 218, 42, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

fn write_test_png(name: &str) -> PathBuf {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test-assets");
    fs::create_dir_all(&directory).expect("test asset directory should be created");
    let path = directory.join(name);
    fs::write(&path, ONE_BY_ONE_PNG).expect("PNG fixture should be written");
    path
}

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

#[test]
fn feature_wraps_sequence_values() {
    let sequence = MLSequence::from_strings(&["cat", "dog"]).expect("sequence should build");
    let feature = Feature::from_sequence(sequence).expect("feature should wrap sequence");
    assert_eq!(feature.feature_type(), FeatureType::Sequence);
    assert_eq!(
        feature.sequence_value().unwrap().string_values(),
        Some(vec!["cat".to_owned(), "dog".to_owned()])
    );
}

#[test]
fn feature_creates_image_values_from_url() {
    let path = write_test_png("one-by-one.png");
    let options = ImageFeatureOptions {
        crop_rect: Some(ImageCropRect {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        }),
        crop_and_scale: Some(ImageCropAndScale::CenterCrop),
    };

    let feature = Feature::from_image_url_with_options(&path, 1, 1, 0x4247_5241, &options)
        .expect("image feature should build from PNG fixture");
    assert_eq!(feature.feature_type(), FeatureType::Image);
}
