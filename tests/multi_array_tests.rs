use std::panic::{catch_unwind, AssertUnwindSafe};

use coreml::prelude::*;
use half::f16;

fn int8_array(shape: &[usize]) -> Option<MultiArray> {
    match MultiArray::new(shape, DataType::Int8) {
        Ok(array) => Some(array),
        Err(CoreMLError::Unsupported(_)) => None,
        Err(error) => panic!("unexpected Int8 allocation error: {error}"),
    }
}

#[test]
fn multi_array_supports_concat_and_number_helpers() {
    let mut left = MultiArray::new_f32(&[2, 2]).expect("left multi-array should allocate");
    left.copy_from_slice(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("left data should copy");
    let mut right = MultiArray::new_f32(&[2, 1]).expect("right multi-array should allocate");
    right
        .copy_from_slice(&[5.0_f32, 6.0])
        .expect("right data should copy");

    let concatenated = MultiArray::concatenate(&[&left, &right], 1, DataType::Float32)
        .expect("concatenation should work");
    assert_eq!(concatenated.shape(), vec![2, 3]);
    assert_eq!(
        concatenated.number_at_indices(&[0, 2]),
        Some(MultiArrayScalar::Float32(5.0))
    );
    assert_eq!(
        concatenated.to_vec::<f32>().unwrap(),
        [1.0, 2.0, 5.0, 3.0, 4.0, 6.0]
    );

    let mut editable = MultiArray::new_i32(&[2]).expect("editable multi-array should allocate");
    editable
        .copy_from_slice(&[1_i32, 2])
        .expect("editable data should copy");
    assert_eq!(
        editable.number_at_linear_index(1),
        Some(MultiArrayScalar::Int32(2))
    );
    editable
        .set_number_at_linear_index(0, 7.5_f64)
        .expect("NSNumber-style setter should cast into Int32 storage");
    assert_eq!(
        editable.number_at_linear_index(0),
        Some(MultiArrayScalar::Int32(7))
    );
    assert_eq!(editable.number_at_linear_index(2), None);
}

#[test]
fn multi_array_transfer_to_is_callable() {
    let mut source = MultiArray::new_f64(&[2, 2]).expect("source multi-array should allocate");
    source
        .copy_from_slice(&[1.0_f64, 2.0, 3.0, 4.0])
        .expect("source data should copy");
    let mut destination =
        MultiArray::new_f32(&[2, 2]).expect("destination multi-array should allocate");

    match source.transfer_to(&mut destination) {
        Ok(()) => assert_eq!(destination.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]),
        Err(CoreMLError::Unsupported(_)) => {}
        Err(error) => panic!("unexpected transfer error: {error}"),
    }
}

#[test]
fn transfer_to_rejects_mismatched_shapes_before_calling_coreml() {
    let source = MultiArray::new_f32(&[2, 2]).unwrap();
    let mut destination = MultiArray::new_f32(&[4]).unwrap();
    let error = source
        .transfer_to(&mut destination)
        .expect_err("shape mismatch must be rejected");
    assert!(matches!(error, CoreMLError::InvalidArgument(_)), "{error}");
}

#[test]
fn new_arrays_are_zero_filled() {
    let array = MultiArray::new_f64(&[3, 5]).unwrap();
    assert_eq!(array.to_vec::<f64>().unwrap(), vec![0.0; 15]);
    array
        .with_bytes(|bytes, strides| {
            assert!(bytes.len() >= 15 * 8);
            assert!(bytes.iter().all(|&byte| byte == 0));
            assert_eq!(strides, [5, 1]);
        })
        .unwrap();
}

#[test]
fn typed_access_rejects_mismatched_element_types() {
    let mut array = MultiArray::new_f32(&[4]).unwrap();
    assert_eq!(array.data_type(), DataType::Float32);
    assert!(matches!(
        array.get::<i32>(&[0]),
        Err(CoreMLError::TypeMismatch(_))
    ));
    assert!(matches!(
        array.set(&[0], 1.0_f64),
        Err(CoreMLError::TypeMismatch(_))
    ));
    assert!(matches!(
        array.with_slice::<f16, _>(|_, _| ()),
        Err(CoreMLError::TypeMismatch(_))
    ));
    assert!(matches!(
        array.with_slice_mut::<i8, _>(|_, _| ()),
        Err(CoreMLError::TypeMismatch(_))
    ));
    assert!(matches!(
        array.to_vec::<f64>(),
        Err(CoreMLError::TypeMismatch(_))
    ));
    assert!(matches!(
        array.copy_from_slice(&[1_i32, 2, 3, 4]),
        Err(CoreMLError::TypeMismatch(_))
    ));
}

#[test]
fn int8_arrays_are_never_read_as_float32() {
    let Some(mut array) = int8_array(&[2, 3]) else {
        return;
    };
    assert_eq!(array.data_type(), DataType::Int8);
    assert_eq!(DataType::Int8.element_size(), Some(1));
    assert!(matches!(
        array.with_slice::<f32, _>(|_, _| ()),
        Err(CoreMLError::TypeMismatch(_))
    ));
    assert!(matches!(
        array.with_slice_mut::<f32, _>(|_, _| ()),
        Err(CoreMLError::TypeMismatch(_))
    ));
    assert!(matches!(
        array.get::<f32>(&[0, 0]),
        Err(CoreMLError::TypeMismatch(_))
    ));
    assert!(matches!(
        array.set(&[1, 2], 1.0_f32),
        Err(CoreMLError::TypeMismatch(_))
    ));

    array
        .copy_from_slice(&[1_i8, -2, 3, -4, 5, -6])
        .expect("Int8 data should copy");
    assert_eq!(array.to_vec::<i8>().unwrap(), [1, -2, 3, -4, 5, -6]);
    assert_eq!(array.get::<i8>(&[1, 2]).unwrap(), -6);
    array
        .with_slice::<i8, _>(|values, strides| {
            assert_eq!(strides, [3, 1]);
            assert!(values.len() >= 6);
            assert_eq!(&values[..6], &[1, -2, 3, -4, 5, -6]);
        })
        .unwrap();

    array.set_number_at_indices(&[0, 0], 1000_i32).unwrap();
    array.set_number_at_indices(&[0, 1], -1000.0_f64).unwrap();
    assert_eq!(
        array.number_at_indices(&[0, 0]),
        Some(MultiArrayScalar::Int8(i8::MAX))
    );
    assert_eq!(
        array.number_at_indices(&[0, 1]),
        Some(MultiArrayScalar::Int8(i8::MIN))
    );
}

#[test]
fn element_access_checks_indices() {
    let mut array = MultiArray::new_f32(&[2, 3]).unwrap();
    assert!(matches!(
        array.get::<f32>(&[2, 0]),
        Err(CoreMLError::IndexOutOfRange(_))
    ));
    assert!(matches!(
        array.get::<f32>(&[0]),
        Err(CoreMLError::IndexOutOfRange(_))
    ));
    assert!(matches!(
        array.set(&[0, 3], 1.0_f32),
        Err(CoreMLError::IndexOutOfRange(_))
    ));
    assert!(array.set_number_at_linear_index(6, 1.0_f32).is_err());
    array.set(&[1, 2], 4.5_f32).unwrap();
    assert_eq!(array.get::<f32>(&[1, 2]).ok(), Some(4.5));
}

#[test]
fn copy_from_slice_takes_the_logical_element_count() {
    let mut array = MultiArray::new_f32(&[2, 3]).unwrap();
    assert!(array.copy_from_slice(&[0.0_f32; 5]).is_err());
    assert!(array.copy_from_slice(&[0.0_f32; 7]).is_err());
    array
        .copy_from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    assert_eq!(array.len(), 6);
    assert_eq!(array.storage_len(), 6);
    assert!(array.is_contiguous());
}

#[test]
fn provider_views_are_shared_reads_and_copies_are_independent() {
    let mut array = MultiArray::new_f32(&[3]).unwrap();
    array.copy_from_slice(&[1.0_f32, 2.0, 3.0]).unwrap();
    let mut provider = FeatureProvider::new();
    provider.insert_multi_array("x", array).unwrap();

    let first = provider.get_multi_array("x").unwrap();
    let second = provider.get_multi_array("x").unwrap();
    assert_eq!(first.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0]);

    let mut copy = second.copy_to_owned().unwrap();
    copy.set(&[0], 9.0_f32).unwrap();
    assert_eq!(copy.to_vec::<f32>().unwrap(), [9.0, 2.0, 3.0]);
    assert_eq!(first.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0]);
    assert_eq!(second.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0]);
    assert!(provider.get_multi_array("missing").is_none());
}

#[test]
fn feature_views_outlive_nothing_and_copy_on_demand() {
    let mut array = MultiArray::new_i32(&[2]).unwrap();
    array.copy_from_slice(&[4_i32, 5]).unwrap();
    let feature = Feature::from_multi_array(array).unwrap();
    let view = feature.multi_array_value().unwrap();
    let copy = view.copy_to_owned().unwrap();
    drop(view);
    drop(feature);
    assert_eq!(copy.to_vec::<i32>().unwrap(), [4, 5]);
}

#[test]
fn unknown_data_types_are_rejected() {
    assert!(matches!(
        MultiArray::new(&[2], DataType::Unknown(7)),
        Err(CoreMLError::InvalidArgument(_))
    ));
    let array = MultiArray::new_f32(&[2]).unwrap();
    assert!(matches!(
        MultiArray::concatenate(&[&array], 0, DataType::Unknown(7)),
        Err(CoreMLError::InvalidArgument(_))
    ));
    assert_eq!(DataType::Unknown(7).element_size(), None);
}

#[test]
fn concatenate_validates_before_calling_coreml() {
    let a = MultiArray::new_f32(&[2, 2]).unwrap();
    let b = MultiArray::new_f32(&[3, 3]).unwrap();
    let c = MultiArray::new_f32(&[2]).unwrap();
    let empty: [&MultiArray; 0] = [];
    assert!(MultiArray::concatenate(&empty, 0, DataType::Float32).is_err());
    assert!(MultiArray::concatenate(&[&a, &b], 1, DataType::Float32).is_err());
    assert!(MultiArray::concatenate(&[&a, &c], 0, DataType::Float32).is_err());
    let joined = MultiArray::concatenate(&[&a, &a], -1, DataType::Float64).unwrap();
    assert_eq!(joined.shape(), [2, 4]);
    assert_eq!(joined.data_type(), DataType::Float64);
}

#[test]
fn panics_inside_storage_closures_propagate_and_leave_the_array_usable() {
    let mut array = MultiArray::new_f32(&[2]).unwrap();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        array
            .with_slice_mut::<f32, _>(|values, _| {
                values[0] = 3.0;
                panic!("user closure panicked");
            })
            .unwrap();
    }));
    assert!(outcome.is_err());
    assert_eq!(array.to_vec::<f32>().unwrap(), [3.0, 0.0]);
}

#[test]
fn float16_values_round_trip() {
    let mut array = MultiArray::new_f16(&[2]).unwrap();
    array
        .copy_from_slice(&[f16::from_f32(0.5), f16::from_f32(-2.0)])
        .unwrap();
    assert_eq!(
        array.number_at_linear_index(1),
        Some(MultiArrayScalar::Float16(f16::from_f32(-2.0)))
    );
    array.set_number_at_linear_index(0, 1.5_f64).unwrap();
    assert_eq!(array.get::<f16>(&[0]).unwrap(), f16::from_f32(1.5));
}
