use coreml::prelude::*;

#[test]
fn multi_array_supports_concat_and_number_helpers() {
    let mut left = MultiArray::new_f32(&[2, 2]).expect("left multi-array should allocate");
    left.copy_from_f32_slice(&[1.0, 2.0, 3.0, 4.0])
        .expect("left data should copy");
    let mut right = MultiArray::new_f32(&[2, 1]).expect("right multi-array should allocate");
    right
        .copy_from_f32_slice(&[5.0, 6.0])
        .expect("right data should copy");

    let concatenated = MultiArray::concatenate(&[&left, &right], 1, DataType::Float32)
        .expect("concatenation should work");
    assert_eq!(concatenated.shape(), vec![2, 3]);
    assert_eq!(
        concatenated.number_at_indices(&[0, 2]),
        Some(MultiArrayScalar::Float32(5.0))
    );

    let mut editable = MultiArray::new_i32(&[2]).expect("editable multi-array should allocate");
    editable
        .copy_from_i32_slice(&[1, 2])
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
}

#[test]
fn multi_array_transfer_to_is_callable() {
    let mut source = MultiArray::new_f64(&[2, 2]).expect("source multi-array should allocate");
    source
        .copy_from_f64_slice(&[1.0, 2.0, 3.0, 4.0])
        .expect("source data should copy");
    let mut destination =
        MultiArray::new_f32(&[2, 2]).expect("destination multi-array should allocate");

    match source.transfer_to(&mut destination) {
        Ok(()) => assert_eq!(
            destination
                .as_f32_slice()
                .expect("destination should be Float32"),
            &[1.0, 2.0, 3.0, 4.0]
        ),
        Err(CoreMLError::Unsupported(_)) => {}
        Err(error) => panic!("unexpected transfer error: {error}"),
    }
}
