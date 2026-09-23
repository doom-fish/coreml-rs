mod common;

use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use coreml::prelude::*;

static TIMEOUT_SETTING: Mutex<()> = Mutex::new(());

#[test]
fn blocking_timeouts_cancel_and_report_timed_out() {
    let _setting = TIMEOUT_SETTING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let source = common::asset_path("sentiment_classifier.mlmodel");
    let compiled = common::compile_model(&source, "blocking-timeout");
    assert_eq!(coreml::blocking_timeout(), None);

    coreml::set_blocking_timeout(Some(Duration::from_nanos(1)));
    for _ in 0..5 {
        let error = ModelStructure::load_from_url(compiled.as_path())
            .expect_err("a 1 ns structure timeout must not report success");
        assert!(
            matches!(error, CoreMLError::TimedOut(_) | CoreMLError::Unsupported(_)),
            "{error}"
        );
        let error = ComputePlan::load_from_url(compiled.as_path(), &ModelConfiguration::new())
            .expect_err("a 1 ns compute-plan timeout must not report success");
        assert!(
            matches!(error, CoreMLError::TimedOut(_) | CoreMLError::Unsupported(_)),
            "{error}"
        );
    }
    thread::sleep(Duration::from_millis(500));

    coreml::set_blocking_timeout(None);
    let structure = ModelStructure::load_from_url(compiled.as_path());
    assert!(
        matches!(structure, Ok(_) | Err(CoreMLError::Unsupported(_))),
        "{structure:?}"
    );
}

#[test]
#[ignore = "CoreML compiles into the per-user temporary directory, outside target/"]
fn timed_out_compiles_discard_their_late_bundles() {
    let _setting = TIMEOUT_SETTING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let (source, stem) = common::unique_model_source("blocking-timeout", "sentiment_classifier");
    coreml::set_blocking_timeout(Some(Duration::from_nanos(1)));
    for _ in 0..5 {
        let error = ModelCompiler::compile(&source)
            .expect_err("a 1 ns compile timeout must not report success");
        assert!(matches!(error, CoreMLError::TimedOut(_)), "{error}");
    }
    coreml::set_blocking_timeout(None);
    thread::sleep(Duration::from_secs(2));
    assert_eq!(common::temporary_bundles(&stem), 0);
}
