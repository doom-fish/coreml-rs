mod common;

use std::cell::RefCell;
use std::thread;
use std::time::Duration;

use coreml::prelude::*;

fn point(x: f64, y: f64) -> MultiArray {
    let mut array = MultiArray::new_f64(&[2]).expect("point should allocate");
    array.copy_from_slice(&[x, y]).expect("point should fill");
    array
}

fn labelled_points(points: &[(f64, f64, &str)]) -> BatchProvider {
    let mut batch = BatchProvider::new();
    for &(x, y, label) in points {
        let mut provider = FeatureProvider::new();
        provider.insert_multi_array("features", point(x, y));
        provider.insert_string("label", label);
        batch.push(provider);
    }
    batch
}

fn predict_label(model: &Model, x: f64, y: f64) -> Option<String> {
    let mut input = FeatureProvider::new();
    input.insert_multi_array("features", point(x, y));
    model
        .predict(&input)
        .expect("prediction should succeed")
        .get_string("label")
}

#[test]
fn update_missing_bundle_fails() {
    let mut provider = FeatureProvider::new();
    provider.insert_double("score", 0.5);
    let batch = BatchProvider::from_feature_providers(vec![provider]);
    let error = Update::run(
        "tests/does-not-exist.mlmodelc",
        &batch,
        None,
        UpdateProgressHandlers::all(),
        None,
    )
    .expect_err("missing update model should fail");
    assert!(matches!(
        error,
        CoreMLError::UpdateFailed(_) | CoreMLError::Unknown { .. }
    ));
}

#[test]
fn update_rejects_models_that_are_not_updatable() {
    let compiled = common::compile_model(
        &common::asset_path("sentiment_classifier.mlmodel"),
        "update-not-updatable",
    );
    let error = Update::run(
        &compiled,
        &labelled_points(&[(0.0, 0.0, "left")]),
        None,
        UpdateProgressHandlers::new(),
        None,
    )
    .expect_err("a non-updatable model must not train");
    assert!(matches!(error, CoreMLError::UpdateFailed(_)), "{error}");
}

#[test]
fn update_returns_the_trained_model_and_delivers_progress_on_the_caller() {
    let compiled = common::compile_model(
        &common::asset_path("updatable_knn.mlmodel"),
        "update-knn",
    );
    let untrained = Model::load_from_url(&compiled, &ModelConfiguration::new())
        .expect("untrained model should load");
    assert_eq!(predict_label(&untrained, 1.0, 1.0).as_deref(), Some("unknown"));

    let caller = thread::current().id();
    let delivered = RefCell::new(Vec::new());
    let handlers = UpdateProgressHandlers::all().on_progress(|context| {
        assert_eq!(thread::current().id(), caller);
        delivered.borrow_mut().push(context.event.clone());
    });
    let batch = labelled_points(&[
        (0.0, 0.0, "left"),
        (0.0, 1.0, "left"),
        (10.0, 10.0, "right"),
        (10.0, 9.0, "right"),
    ]);
    let outcome = Update::run(
        &compiled,
        &batch,
        None,
        handlers,
        Some(Duration::from_secs(120)),
    )
    .expect("the k-NN update should complete");

    assert_eq!(outcome.result.final_state, UpdateTaskState::Completed);
    let completion = outcome.result.contexts.last().expect("completion context");
    assert_eq!(completion.event, "completion");
    let progress_events: Vec<String> = outcome.result.contexts[..outcome.result.contexts.len() - 1]
        .iter()
        .map(|context| context.event.clone())
        .collect();
    let delivered = delivered.into_inner();
    assert!(
        delivered.iter().any(|event| event == "training_begin"),
        "{delivered:?}"
    );
    assert_eq!(delivered, progress_events);

    assert_eq!(predict_label(&outcome.model, 1.0, 1.0).as_deref(), Some("left"));
    assert_eq!(predict_label(&outcome.model, 9.0, 9.0).as_deref(), Some("right"));
    assert_eq!(predict_label(&untrained, 9.0, 9.0).as_deref(), Some("unknown"));

    let saved = common::artifact_dir("update-knn").join("trained.mlmodelc");
    if saved.exists() {
        std::fs::remove_dir_all(&saved).expect("remove stale trained model");
    }
    outcome
        .model
        .write_to_url(&saved)
        .expect("the updated model should be writable");
    let reloaded = Model::load_from_url(&saved, &ModelConfiguration::new())
        .expect("the written model should load");
    assert_eq!(predict_label(&reloaded, 0.5, 0.5).as_deref(), Some("left"));
}

#[test]
fn update_timeouts_cancel_and_never_report_success() {
    let compiled = common::compile_model(
        &common::asset_path("updatable_knn.mlmodel"),
        "update-knn-timeout",
    );
    let points: Vec<(f64, f64, &str)> = (0..2_000)
        .map(|index| {
            let value = f64::from(index);
            (value, value, if index % 2 == 0 { "even" } else { "odd" })
        })
        .collect();
    let batch = labelled_points(&points);
    let result = Update::run(
        &compiled,
        &batch,
        None,
        UpdateProgressHandlers::all(),
        Some(Duration::ZERO),
    );
    match result {
        Err(CoreMLError::TimedOut(_)) => {}
        Ok(outcome) => assert_eq!(outcome.result.final_state, UpdateTaskState::Completed),
        Err(error) => panic!("unexpected update error: {error}"),
    }
    thread::sleep(Duration::from_millis(500));
}
