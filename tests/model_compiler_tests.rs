mod common;

use coreml::prelude::*;

#[test]
fn model_compiler_missing_source_fails() {
    let error = ModelCompiler::compile("tests/does-not-exist.mlmodel")
        .expect_err("missing .mlmodel source should fail");
    assert!(matches!(
        error,
        CoreMLError::CompilationFailed(_) | CoreMLError::Unknown { .. }
    ));
}

#[test]
#[ignore = "CoreML compiles into the per-user temporary directory, outside target/"]
fn compile_and_load_removes_its_temporary_bundle_with_the_model() {
    let (source, stem) = common::unique_model_source("compiler-cleanup", "sentiment_classifier");
    assert_eq!(common::temporary_bundles(&stem), 0);

    let model = ModelCompiler::compile_and_load(&source, &ModelConfiguration::new())
        .expect("compile and load should succeed");
    assert_eq!(common::temporary_bundles(&stem), 1);
    let mut inputs = FeatureProvider::new();
    inputs.insert_string("text", "I love this product");
    assert_eq!(
        model.predict(&inputs).unwrap().get_string("label").as_deref(),
        Some("positive")
    );
    drop(model);
    assert_eq!(common::temporary_bundles(&stem), 0);

    let compiled = ModelCompiler::compile(&source).expect("plain compile should succeed");
    assert_eq!(common::temporary_bundles(&stem), 1);
    std::fs::remove_dir_all(compiled).expect("the caller owns a plain compile result");
    assert_eq!(common::temporary_bundles(&stem), 0);
}

#[cfg(feature = "async")]
#[test]
#[ignore = "CoreML compiles into the per-user temporary directory, outside target/"]
fn compile_and_load_async_removes_its_temporary_bundle_with_the_model() {
    let (source, stem) =
        common::unique_model_source("compiler-cleanup-async", "sentiment_classifier");
    let model = pollster::block_on(ModelCompiler::compile_and_load_async(&source, None))
        .expect("async compile and load should succeed");
    assert_eq!(common::temporary_bundles(&stem), 1);
    drop(model);
    assert_eq!(common::temporary_bundles(&stem), 0);
}
