use coreml::prelude::*;

struct ScaleAndBiasLayer;

impl MLCustomLayer for ScaleAndBiasLayer {
    fn output_shapes_for_input_shapes(
        &self,
        input_shapes: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>, CoreMLError> {
        Ok(input_shapes.to_vec())
    }

    fn evaluate_on_cpu(
        &mut self,
        _inputs: &[MultiArray],
        _outputs: &mut [MultiArray],
    ) -> Result<(), CoreMLError> {
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registration =
        MLCustomLayerRegistration::register("RustExampleScaleAndBiasLayer", |_context| {
            Ok(ScaleAndBiasLayer)
        })?;
    println!("registered custom layer class: {}", registration.class_name());
    Ok(())
}
