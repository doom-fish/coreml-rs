#![cfg(feature = "async")]

mod common;

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Wake, Waker};
use std::thread;
use std::time::Duration;

use coreml::prelude::*;

struct NoopWake;

impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

fn poll_once<F: Future>(future: F) {
    let waker = Waker::from(Arc::new(NoopWake));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    let _ = future.as_mut().poll(&mut context);
}

#[test]
fn dropped_async_futures_cancel_without_touching_rust_memory() {
    let source = common::asset_path("sentiment_classifier.mlmodel");
    let compiled = common::compile_model(&source, "async-cancel");
    let model = pollster::block_on(Model::load_async(compiled.as_path(), None))
        .expect("model should load");

    let mut inputs = FeatureProvider::new();
    inputs.insert_string("text", "I love this product").unwrap();
    for round in 0..20 {
        poll_once(model.predict_async(&inputs, None));
        poll_once(Model::load_async(compiled.as_path(), None));
        inputs.insert_string("text", if round % 2 == 0 { "awful" } else { "I love this product" }).unwrap();
    }
    thread::sleep(Duration::from_millis(500));

    inputs.insert_string("text", "I love this product").unwrap();
    let outputs = pollster::block_on(model.predict_async(&inputs, None))
        .expect("prediction after cancelled futures should succeed");
    assert_eq!(outputs.get_string("label").as_deref(), Some("positive"));
}

#[test]
#[ignore = "CoreML compiles into the per-user temporary directory, outside target/"]
fn dropped_compile_futures_remove_late_bundles() {
    let (source, stem) = common::unique_model_source("async-cancel", "sentiment_classifier");
    for _ in 0..10 {
        poll_once(ModelCompiler::compile_async(source.as_path()));
    }
    thread::sleep(Duration::from_secs(2));
    assert_eq!(common::temporary_bundles(&stem), 0);
}
