use coreml::ml_state::MLState;

fn main() {
    println!(
        "mlstate runtime supported: {}",
        MLState::runtime_supported()
    );
}
