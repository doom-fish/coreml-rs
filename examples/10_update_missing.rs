use coreml::prelude::*;

fn main() {
    let mut provider = FeatureProvider::new();
    provider.insert_double("score", 0.5).expect("feature should insert");
    let batch = BatchProvider::from_feature_providers(vec![provider]);
    let handlers = UpdateProgressHandlers::all();
    let error = Update::run(
        "examples/does-not-exist.mlmodelc",
        &batch,
        None,
        handlers,
        None,
    )
    .expect_err("running update on a missing model should fail");
    println!("update error: {error}");
}
