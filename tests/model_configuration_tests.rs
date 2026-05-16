use coreml::prelude::*;

#[test]
fn model_configuration_bridge_snapshot_round_trips() {
    let configuration = ModelConfiguration::new()
        .with_compute_units(ComputeUnits::CpuAndNeuralEngine)
        .with_allow_low_precision_accumulation_on_gpu(true)
        .with_display_name("integration test")
        .with_function_name("main")
        .with_optimization_hints(OptimizationHints {
            reshape_frequency: ReshapeFrequencyHint::Infrequent,
            specialization_strategy: Some(SpecializationStrategy::FastPrediction),
        })
        .with_parameter("epochs", 2_i64)
        .with_parameter("learning_rate", 0.01_f64);

    let snapshot = configuration
        .bridge_snapshot()
        .expect("configuration should round-trip through Swift");
    assert_eq!(snapshot.compute_units(), ComputeUnits::CpuAndNeuralEngine);
    assert!(snapshot.allow_low_precision_accumulation_on_gpu());
    assert_eq!(snapshot.display_name(), Some("integration test"));
    assert_eq!(
        snapshot.parameters().get("epochs"),
        Some(&ParameterValue::Int64(2))
    );
}
