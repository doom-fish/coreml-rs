mod common;

use std::process::Command;

use coreml::ml_state::MLState;
use coreml::prelude::*;

fn major_version(program: &str, args: &[&str]) -> u32 {
    let output = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("failed to run {program}: {error}"));
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .split('.')
        .next()
        .and_then(|major| major.parse().ok())
        .unwrap_or_else(|| panic!("{program} printed no version"))
}

#[test]
fn ml_state_runtime_support_matches_the_os_and_sdk() {
    let os_major = major_version("sw_vers", &["-productVersion"]);
    let sdk_major = major_version("xcrun", &["--sdk", "macosx", "--show-sdk-version"]);
    assert_eq!(
        MLState::runtime_supported(),
        os_major >= 15 && sdk_major >= 15
    );
}

#[test]
fn stateless_models_do_not_vend_state() {
    let source = common::asset_path("sentiment_classifier.mlmodel");
    let compiled = common::compile_model(&source, "ml-state");
    let model = Model::load_from_url(&compiled, &ModelConfiguration::new())
        .expect("compiled fixture should load");
    let error = model
        .new_state()
        .expect_err("a model without state features has no MLState");
    assert!(
        matches!(
            error,
            CoreMLError::StateFailed(_) | CoreMLError::Unsupported(_)
        ),
        "{error}"
    );
}
