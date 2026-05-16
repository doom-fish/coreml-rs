use coreml::prelude::*;

fn main() {
    let configuration = ModelConfiguration::new().with_compute_units(ComputeUnits::CpuOnly);
    let error = Model::load_from_url("examples/does-not-exist.mlmodelc", &configuration)
        .expect_err("loading a missing model should fail");
    println!("missing-model error: {error}");
}
