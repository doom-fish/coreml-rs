use std::collections::BTreeMap;

use coreml::prelude::*;
use serde_json::{json, Value};

struct AffineLayer {
    scale: f32,
    bias: f32,
}

impl MLCustomLayer for AffineLayer {
    fn set_weight_data(&mut self, weights: &[Vec<u8>]) -> Result<(), CoreMLError> {
        if let Some(bytes) = weights.first() {
            if bytes.len() < 4 {
                return Err(CoreMLError::CustomLayerFailed(
                    "expected at least four bytes for the bias weight".to_owned(),
                ));
            }
            let mut raw = [0_u8; 4];
            raw.copy_from_slice(&bytes[..4]);
            self.bias = f32::from_le_bytes(raw);
        }
        Ok(())
    }

    fn output_shapes_for_input_shapes(
        &self,
        input_shapes: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>, CoreMLError> {
        Ok(input_shapes.to_vec())
    }

    fn evaluate_on_cpu(
        &mut self,
        inputs: &[&MultiArrayRef],
        outputs: &mut [&mut MultiArrayRef],
    ) -> Result<(), CoreMLError> {
        let [input] = inputs else {
            return Err(CoreMLError::CustomLayerFailed(
                "expected one Float32 input tensor".to_owned(),
            ));
        };
        let [output] = outputs else {
            return Err(CoreMLError::CustomLayerFailed(
                "expected one Float32 output tensor".to_owned(),
            ));
        };
        let values: Vec<f32> = input
            .to_vec::<f32>()?
            .into_iter()
            .map(|value| value.mul_add(self.scale, self.bias))
            .collect();
        output.copy_from_slice(&values)
    }
}

struct PanickingLayer;

impl MLCustomLayer for PanickingLayer {
    fn output_shapes_for_input_shapes(
        &self,
        input_shapes: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>, CoreMLError> {
        Ok(input_shapes.to_vec())
    }

    fn evaluate_on_cpu(
        &mut self,
        _inputs: &[&MultiArrayRef],
        outputs: &mut [&mut MultiArrayRef],
    ) -> Result<(), CoreMLError> {
        outputs[0].set(&[0], 42.0_f32)?;
        panic!("custom layer exploded");
    }
}

fn layer_parameters(scale: f32) -> BTreeMap<String, Value> {
    BTreeMap::from([(String::from("scale"), json!(scale))])
}

#[test]
fn custom_layer_registration_round_trips_cpu_callbacks() {
    let registration = MLCustomLayerRegistration::register("RustTestAffineLayer", |context| {
        let scale = context
            .parameters
            .get("scale")
            .cloned()
            .and_then(|value| serde_json::from_value::<f32>(value).ok())
            .unwrap_or(1.0);
        Ok(AffineLayer { scale, bias: 0.0 })
    })
    .expect("custom layer should register");

    let mut layer = registration
        .instantiate(&layer_parameters(2.0))
        .expect("custom layer should instantiate");
    layer
        .set_weight_data(&[1.5_f32.to_le_bytes().to_vec()])
        .expect("custom layer should accept weights");

    let shapes = layer
        .output_shapes_for_input_shapes(&[vec![3]])
        .expect("custom layer should compute output shapes");
    assert_eq!(shapes, vec![vec![3]]);

    let mut input = MultiArray::new_f32(&[3]).expect("input tensor should allocate");
    input
        .copy_from_slice(&[1.0_f32, 2.0, -1.0])
        .expect("input tensor should fill");
    let mut output = MultiArray::new_f32(&[3]).expect("output tensor should allocate");
    layer
        .evaluate_on_cpu(&[&input], &mut [&mut output])
        .expect("custom layer should run on CPU");

    assert_eq!(output.to_vec::<f32>().unwrap(), [3.5, 5.5, -0.5]);
}

#[test]
fn custom_layer_registration_rejects_duplicate_class_names() {
    let _registration = MLCustomLayerRegistration::register("RustDuplicateLayer", |_context| {
        Ok(AffineLayer {
            scale: 1.0,
            bias: 0.0,
        })
    })
    .expect("first custom layer registration should succeed");

    let error = MLCustomLayerRegistration::register("RustDuplicateLayer", |_context| {
        Ok(AffineLayer {
            scale: 1.0,
            bias: 0.0,
        })
    })
    .expect_err("duplicate custom layer registrations must fail");

    assert!(matches!(error, CoreMLError::InvalidArgument(_)));
}

#[test]
fn custom_layer_panics_become_errors_without_releasing_lent_arrays() {
    let registration =
        MLCustomLayerRegistration::register("RustPanickingLayer", |_context| Ok(PanickingLayer))
            .expect("custom layer should register");
    let mut layer = registration
        .instantiate(&BTreeMap::new())
        .expect("custom layer should instantiate");

    let input = MultiArray::new_f32(&[2]).expect("input tensor should allocate");
    let mut output = MultiArray::new_f32(&[2]).expect("output tensor should allocate");
    for _ in 0..3 {
        let error = layer
            .evaluate_on_cpu(&[&input], &mut [&mut output])
            .expect_err("a panicking layer must report an error");
        assert!(matches!(error, CoreMLError::CustomLayerFailed(_)), "{error}");
    }
    drop(layer);

    assert_eq!(output.to_vec::<f32>().unwrap(), [42.0, 0.0]);
    assert_eq!(input.to_vec::<f32>().unwrap(), [0.0, 0.0]);
    let copy = output.copy_to_owned().unwrap();
    drop(output);
    assert_eq!(copy.to_vec::<f32>().unwrap(), [42.0, 0.0]);
}

struct DropPanickingLayer;

impl MLCustomLayer for DropPanickingLayer {
    fn output_shapes_for_input_shapes(
        &self,
        input_shapes: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>, CoreMLError> {
        Ok(input_shapes.to_vec())
    }

    fn evaluate_on_cpu(
        &mut self,
        _inputs: &[&MultiArrayRef],
        _outputs: &mut [&mut MultiArrayRef],
    ) -> Result<(), CoreMLError> {
        Ok(())
    }
}

impl Drop for DropPanickingLayer {
    fn drop(&mut self) {
        panic!("custom layer drop exploded");
    }
}

#[test]
fn custom_layer_panics_while_dropping_are_contained() {
    let registration =
        MLCustomLayerRegistration::register("RustDropPanickingLayer", |_context| Ok(DropPanickingLayer))
            .expect("custom layer should register");
    for _ in 0..2 {
        let layer = registration
            .instantiate(&BTreeMap::new())
            .expect("custom layer should instantiate");
        drop(layer);
    }
}

#[test]
fn custom_layer_input_shapes_that_do_not_fit_in_int_are_errors() {
    let registration = MLCustomLayerRegistration::register("RustShapeCheckingLayer", |_context| {
        Ok(AffineLayer {
            scale: 1.0,
            bias: 0.0,
        })
    })
    .expect("custom layer should register");
    let layer = registration
        .instantiate(&BTreeMap::new())
        .expect("custom layer should instantiate");
    let error = layer
        .output_shapes_for_input_shapes(&[vec![usize::MAX]])
        .expect_err("a dimension above Int.max must not become an empty shape list");
    assert!(matches!(error, CoreMLError::InvalidArgument(_)), "{error}");
    assert_eq!(layer.output_shapes_for_input_shapes(&[vec![2, 3]]).unwrap(), [vec![2, 3]]);
}
