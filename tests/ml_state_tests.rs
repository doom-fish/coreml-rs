use coreml::ml_state::MLState;

#[test]
fn ml_state_runtime_supported_is_queryable() {
    let value = MLState::runtime_supported();
    assert!(matches!(value, true | false));
}
