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
