use coreml::prelude::*;

fn main() {
    let error = ComputePlan::load_from_url(
        "examples/does-not-exist.mlmodelc",
        &ModelConfiguration::new(),
    )
    .expect_err("loading a compute plan for a missing model should fail");
    let details_error = ComputePlan::load_details_from_url(
        "examples/does-not-exist.mlmodelc",
        &ModelConfiguration::new(),
    )
    .expect_err("loading detailed compute-plan data for a missing model should fail");
    println!("compute-plan error: {error}");
    println!("compute-plan details error: {details_error}");
}
