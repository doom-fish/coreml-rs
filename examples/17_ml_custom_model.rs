use coreml::prelude::*;

struct ScaleModel;

impl MLCustomModel for ScaleModel {
    fn prediction_from_features(
        &mut self,
        _input: &FeatureProvider,
        _options: &PredictionOptions,
    ) -> Result<FeatureProvider, CoreMLError> {
        Ok(FeatureProvider::new())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registration =
        MLCustomModelRegistration::register("RustExampleScaleModel", |_context| Ok(ScaleModel))?;
    println!(
        "registered custom model class: {}",
        registration.class_name()
    );
    Ok(())
}
