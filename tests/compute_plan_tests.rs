use coreml::prelude::*;

#[test]
fn compute_plan_missing_bundle_fails() {
    let error =
        ComputePlan::load_from_url("tests/does-not-exist.mlmodelc", &ModelConfiguration::new())
            .expect_err("missing compute-plan model should fail");
    assert!(matches!(
        error,
        CoreMLError::ComputePlanFailed(_)
            | CoreMLError::Unsupported(_)
            | CoreMLError::Unknown { .. }
    ));

    let details_error = ComputePlan::load_details_from_url(
        "tests/does-not-exist.mlmodelc",
        &ModelConfiguration::new(),
    )
    .expect_err("missing compute-plan model should fail for detailed inspection too");
    assert!(matches!(
        details_error,
        CoreMLError::ComputePlanFailed(_)
            | CoreMLError::Unsupported(_)
            | CoreMLError::Unknown { .. }
    ));
}
