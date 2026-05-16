use coreml::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut array = MultiArray::new_f32(&[2, 3])?;
    array.copy_from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    assert_eq!(array.shape(), vec![2, 3]);
    assert_eq!(array.strides(), vec![3, 1]);
    assert_eq!(array.get_f32(&[1, 2]), Some(6.0));

    array.set_f32(&[0, 1], 9.0)?;
    assert_eq!(
        array.as_f32_slice().unwrap(),
        &[1.0, 9.0, 3.0, 4.0, 5.0, 6.0]
    );

    let expected_tensor = array.as_f32_slice().unwrap().to_vec();

    let mut inputs = FeatureProvider::new();
    inputs.insert_multi_array("tensor", array);
    inputs.insert_string("label", "doom fish");
    inputs.insert_int64("count", 42);
    inputs.insert_double("score", 0.5);

    assert_eq!(
        inputs.keys(),
        vec![
            "count".to_owned(),
            "label".to_owned(),
            "score".to_owned(),
            "tensor".to_owned(),
        ]
    );
    assert_eq!(inputs.get_string("label").as_deref(), Some("doom fish"));
    assert_eq!(inputs.get_int64("count"), Some(42));
    assert_eq!(inputs.get_double("score"), Some(0.5));

    let round_trip = inputs
        .get_multi_array("tensor")
        .expect("tensor should exist");
    assert_eq!(round_trip.shape(), vec![2, 3]);
    assert_eq!(
        round_trip.as_f32_slice().unwrap(),
        expected_tensor.as_slice()
    );

    let configuration = ModelConfiguration::new()
        .with_compute_units(ComputeUnits::CpuOnly)
        .with_display_name("coreml smoke");
    let load_error = Model::load_from_url("examples/does-not-exist.mlmodelc", &configuration)
        .expect_err("loading a missing model should fail");
    println!("missing-model error: {load_error}");

    println!("✅ coreml smoke OK");
    Ok(())
}
