# coreml

Safe, idiomatic Rust bindings for Apple’s [CoreML](https://developer.apple.com/documentation/coreml) framework — load models, inspect descriptions, build feature values/providers/batches, register Rust-backed custom layers/models, request compute-plan summaries, run model updates, and use stateful inference on macOS.

## Features

- **Model** — load compiled `.mlmodelc` bundles, load in-memory specifications via `MLModelAsset`, and run synchronous prediction APIs.
- **ModelDescription** — decode rich snapshots covering input/output/state/training features, metadata, and parameter constraints.
- **Feature** — construct `MLFeatureValue` wrappers for integers, doubles, strings, multi-arrays, undefined values, and string/int keyed dictionaries.
- **Prediction** — configure `PredictionOptions` and round-trip them through the Swift bridge.
- **ModelConfiguration** — configure compute units, low-precision GPU accumulation, display/function names, optimization hints, and parameter dictionaries.
- **ComputePlan** — request `MLComputePlan` summaries for compiled models.
- **ModelCompiler** — compile source `.mlmodel` files into temporary `.mlmodelc` bundles.
- **BatchProvider / MLArrayBatchProvider** — build CoreML batches from feature-provider arrays.
- **Update** — run `MLUpdateTask` workflows, receive progress callbacks on the calling thread, and get the updated model back.
- **MLDictionaryFeatureProvider** — build mutable dictionary-backed feature providers.
- **MLState** — create `MLState` handles, run serialized stateful predictions, and read, write, or snapshot named state buffers.
- **MLCustomLayer / MLCustomModel** — register Rust callback implementations as Objective-C CoreML custom layers/models and exercise them in headless tests/examples.
- **MultiArray** — allocate and mutate `MLMultiArray` tensors with `Float32`, `Float16`, `Int32`, `Float64`, and (macOS 26+) `Int8` storage through scoped, type-checked access.

## Requirements

- macOS 13.0 or newer. Some APIs need a newer system and return `CoreMLError::Unsupported` on older ones: async prediction (14.0), `ComputePlan` and `ModelStructure` (14.4), `MLState` and `MultiArray::transfer_to` (15.0), and `Int8` multi-arrays (26.0).
- Xcode 15+ with the macOS SDK
- A compiled `.mlmodelc` bundle, a source `.mlmodel`, or a `.mlmodel` specification in memory

## Installation

```toml
[dependencies]
coreml = "0.4.0"
```

## Async API

Enable the optional `async` feature for executor-agnostic futures around model
loading/compilation and one-shot prediction:

```toml
[dependencies]
coreml = { version = "0.4.0", features = ["async"] }
```

This covers `Model::load_async`, `ModelCompiler::compile_async`,
`Model::compile_model_async`, `Model::compile_and_load_async`,
`Model::predict_async`, and `Model::predict_with_state_async`.

## Quick start

```rust,no_run
use coreml::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let configuration = ModelConfiguration::new().with_compute_units(ComputeUnits::All);
    let model = Model::load_from_url("MyModel.mlmodelc", &configuration)?;

    let mut tensor = MultiArray::new_f32(&[1, 3, 224, 224])?;
    let pixels = vec![0.0_f32; tensor.len()];
    tensor.copy_from_slice(&pixels)?;

    let mut inputs = FeatureProvider::new();
    inputs.insert_multi_array("image", tensor)?;

    let options = PredictionOptions::new().with_uses_cpu_only(false);
    let outputs = model.predict_with_options(&inputs, &options)?;
    println!("output keys: {:?}", outputs.keys());
    Ok(())
}
```

## Multi-array access

`MultiArray` owns an `MLMultiArray` and dereferences to `MultiArrayRef`, which
carries every accessor. Arrays read from a `FeatureProvider` or `Feature` come
back as a read-only `MultiArrayView` tied to that borrow; call `copy_to_owned`
for a mutable copy. Element access (`get`, `set`, `to_vec`, `copy_from_slice`,
`with_slice`, `with_bytes` and their `_mut` forms) runs inside CoreML's scoped
`getBytesWithHandler` / `getMutableBytesWithHandler` blocks, checks the element
type, bounds-checks every offset, and follows the array's strides. Storage
slices never outlive the closure they are handed to.

## Blocking calls, timeouts and cancellation

The synchronous wrappers over CoreML's async APIs (`ModelCompiler::compile`,
`Model::load_from_specification_data`, `ComputePlan`, `ModelStructure`) wait
until CoreML finishes. `coreml::set_blocking_timeout(Some(duration))` bounds
them: on timeout the CoreML task is cancelled and `CoreMLError::TimedOut` is
returned. `Update::run` takes its own timeout. Dropping an async future cancels
its CoreML task. Stateful predictions take `&mut MLState`, and the bridge also
serializes predictions and buffer access per state.

## Examples

The crate ships with 17 headless examples:

- `01_smoke`
- `02_model_load_missing`
- `03_model_description_snapshot`
- `04_feature_values`
- `05_prediction_options`
- `06_model_configuration`
- `07_compute_plan_missing`
- `08_model_compiler_missing`
- `09_batch_provider`
- `10_update_missing`
- `11_ml_dictionary_feature_provider`
- `12_ml_array_batch_provider`
- `13_ml_state_support`
- `14_compute_devices`
- `15_model_structure_snapshot`
- `16_ml_custom_layer`
- `17_ml_custom_model`

Run one example with:

```bash
cargo run --example 06_model_configuration
```

## Coverage notes

See [COVERAGE.md](COVERAGE.md) and [COVERAGE_AUDIT.md](COVERAGE_AUDIT.md) for the SDK audit. The audit counts the 92 top-level Objective-C CoreML symbols (classes, protocols, enums, constants and functions), not individual methods, so its 100% figure does not mean every method is wrapped. Not wrapped: `MLTensor` and `MLShapedArray` (Swift-only), `MLPredictionOptions.outputBackings`, image features from `CGImage` or `CVPixelBuffer` conversion, and the custom-stride, data-pointer and pixel-buffer `MLMultiArray` initializers.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.
