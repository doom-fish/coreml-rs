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
        inputs: &[MultiArray],
        outputs: &mut [MultiArray],
    ) -> Result<(), CoreMLError> {
        let input = inputs
            .first()
            .and_then(MultiArray::as_f32_slice)
            .ok_or_else(|| {
                CoreMLError::CustomLayerFailed("expected one Float32 input tensor".to_owned())
            })?;
        let output = outputs
            .first_mut()
            .and_then(MultiArray::as_f32_slice_mut)
            .ok_or_else(|| {
                CoreMLError::CustomLayerFailed("expected one Float32 output tensor".to_owned())
            })?;
        for (dst, src) in output.iter_mut().zip(input) {
            *dst = (*src).mul_add(self.scale, self.bias);
        }
        Ok(())
    }
}

fn layer_parameters(scale: f32) -> BTreeMap<String, Value> {
    BTreeMap::from([(String::from("scale"), json!(scale))])
}

#[test]
#[ignore = "run via examples/16_ml_custom_layer.rs; direct integration-test worker threads crash CoreML custom callbacks"]
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
        .copy_from_f32_slice(&[1.0, 2.0, -1.0])
        .expect("input tensor should fill");
    let mut output = MultiArray::new_f32(&[3]).expect("output tensor should allocate");
    layer
        .evaluate_on_cpu(
            std::slice::from_ref(&input),
            std::slice::from_mut(&mut output),
        )
        .expect("custom layer should run on CPU");

    assert_eq!(output.as_f32_slice(), Some(&[3.5, 5.5, -0.5][..]));
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
