# coreml

Safe, idiomatic Rust bindings for Apple’s [CoreML](https://developer.apple.com/documentation/coreml) framework — load models, inspect descriptions, build feature values/providers/batches, request compute-plan summaries, run model updates, and use stateful inference on macOS.

## Features

- **Model** — load compiled `.mlmodelc` bundles, load in-memory specifications via `MLModelAsset`, and run synchronous prediction APIs.
- **ModelDescription** — decode rich snapshots covering input/output/state/training features, metadata, and parameter constraints.
- **Feature** — construct `MLFeatureValue` wrappers for integers, doubles, strings, multi-arrays, undefined values, and string/int keyed dictionaries.
- **Prediction** — configure `PredictionOptions` and round-trip them through the Swift bridge.
- **ModelConfiguration** — configure compute units, low-precision GPU accumulation, display/function names, optimization hints, and parameter dictionaries.
- **ComputePlan** — request `MLComputePlan` summaries for compiled models.
- **ModelCompiler** — compile source `.mlmodel` files into temporary `.mlmodelc` bundles.
- **BatchProvider / MLArrayBatchProvider** — build CoreML batches from feature-provider arrays.
- **Update** — run `MLUpdateTask` workflows synchronously and capture progress/completion contexts.
- **MLDictionaryFeatureProvider** — build mutable dictionary-backed feature providers.
- **MLState** — create `MLState` handles, run stateful predictions, and snapshot named state buffers.
- **MultiArray** — allocate and mutate `MLMultiArray` tensors with `Float32`, `Float16`, `Int32`, and `Float64` storage.

## Requirements

- macOS 13.0 or newer
- Xcode 15+ with the macOS SDK
- A compiled `.mlmodelc` bundle, a source `.mlmodel`, or a `.mlmodel` specification in memory

## Installation

```toml
[dependencies]
coreml = "0.2.0"
```

## Quick start

```rust,no_run
use coreml::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let configuration = ModelConfiguration::new().with_compute_units(ComputeUnits::All);
    let model = Model::load_from_url("MyModel.mlmodelc", &configuration)?;

    let mut tensor = MultiArray::new_f32(&[1, 3, 224, 224])?;
    tensor.copy_from_f32_slice(&vec![0.0; tensor.len()])?;

    let mut inputs = FeatureProvider::new();
    inputs.insert_multi_array("image", tensor);

    let options = PredictionOptions::new().with_uses_cpu_only(false);
    let outputs = model.predict_with_options(&inputs, &options)?;
    println!("output keys: {:?}", outputs.keys());
    Ok(())
}
```

## Examples

The crate ships with 13 headless examples:

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

Run one example with:

```bash
cargo run --example 06_model_configuration
```

## Coverage notes

See [COVERAGE.md](COVERAGE.md) for the header audit. The main deferred surfaces in `v0.2.0` are:

- Detailed `MLComputePlanCost` / `MLComputePlanDeviceUsage` objects and the full `MLModelStructure*` graph.
- Compute-device discovery APIs (`MLAllComputeDevices`, `MLCPUComputeDevice`, `MLGPUComputeDevice`, `MLNeuralEngineComputeDevice`).
- Safe wrappers for `MLSequence` values and `MLFeatureValue+MLImageConversion` helpers.
- Remote model-collection management (`MLModelCollection*`) and custom model/layer authoring APIs.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.
