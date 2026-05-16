use coreml::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let configuration = ModelConfiguration::new()
        .with_compute_units(ComputeUnits::CpuAndNeuralEngine)
        .with_allow_low_precision_accumulation_on_gpu(true)
        .with_display_name("coreml example")
        .with_function_name("main")
        .with_optimization_hints(OptimizationHints {
            reshape_frequency: ReshapeFrequencyHint::Infrequent,
            specialization_strategy: Some(SpecializationStrategy::FastPrediction),
        })
        .with_parameter("epochs", 3_i64)
        .with_parameter("learning_rate", 0.01_f64);

    let snapshot = configuration.bridge_snapshot()?;
    assert_eq!(snapshot.compute_units(), ComputeUnits::CpuAndNeuralEngine);
    assert!(snapshot.allow_low_precision_accumulation_on_gpu());
    assert_eq!(snapshot.display_name(), Some("coreml example"));
    println!("configuration parameters: {:?}", snapshot.parameters());
    Ok(())
}
