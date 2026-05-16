use coreml::prelude::*;

#[test]
fn prediction_options_bridge_snapshot_round_trips() {
    let options = PredictionOptions::new().with_uses_cpu_only(true);
    let snapshot = options
        .bridge_snapshot()
        .expect("prediction options should round-trip through Swift");
    assert!(snapshot.uses_cpu_only());
}
