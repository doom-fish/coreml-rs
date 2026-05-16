use coreml::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = PredictionOptions::new().with_uses_cpu_only(true);
    let snapshot = options.bridge_snapshot()?;
    assert!(snapshot.uses_cpu_only());
    println!("prediction uses_cpu_only = {}", snapshot.uses_cpu_only());
    Ok(())
}
