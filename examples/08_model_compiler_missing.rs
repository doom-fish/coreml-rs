use coreml::prelude::*;

fn main() {
    let error = ModelCompiler::compile("examples/does-not-exist.mlmodel")
        .expect_err("compiling a missing model should fail");
    println!("compiler error: {error}");
}
