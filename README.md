# coreml

Safe, idiomatic Rust bindings for Apple's [CoreML](https://developer.apple.com/documentation/coreml) framework — load compiled models, inspect model descriptions, and run on-device inference on macOS.

## Features

- **Compiled model loading** — `Model::load_from_url("MyModel.mlmodelc", ...)`
- **Runtime compilation** — `Model::compile_model` and `Model::compile_and_load` for `.mlmodel` sources
- **In-memory loading** — `Model::load_from_specification_data` uses `MLModelAsset` under the hood
- **Tensor inputs and outputs** — `MultiArray` supports `Float32`, `Float16`, `Int32`, and `Float64`
- **Dictionary feature providers** — build named model inputs from tensors, strings, integers, doubles, and `CVPixelBuffer`s
- **Batch prediction** — `BatchProvider` wraps `MLArrayBatchProvider`
- **Model introspection** — inspect input/output feature descriptions, metadata, and image / tensor constraints
- **Compute-unit selection** — CPU-only, CPU+GPU, CPU+Neural Engine, or `All`

## Requirements

- macOS 13.0 or newer
- Xcode 15+ with the macOS SDK
- A compiled `.mlmodelc` bundle, a source `.mlmodel`, or a `.mlmodel` specification in memory

## Installation

```toml
[dependencies]
coreml = "0.1.0"
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

    let outputs = model.predict(&inputs)?;
    println!("output keys: {:?}", outputs.keys());
    Ok(())
}
```

## Smoke example

```bash
cargo run --example 01_smoke
```

The smoke example does not require a real model. It verifies:

- `MultiArray` creation, shape/stride access, and round-trip element writes
- `FeatureProvider` insert / get for tensors, strings, integers, and doubles
- clean error propagation when loading a nonexistent `.mlmodelc`

## Notes

- Image inputs are exposed via `apple-cf`'s `CVPixelBuffer` wrapper to stay aligned with the rest of the doom-fish macOS stack.
- `MLUpdateTask`, `MLComputePlan`, and macOS 15 stateful inference APIs are intentionally deferred to a future release.
- `preferredMetalDevice` and typed `MLParameterKey` dictionaries are not yet surfaced in the safe Rust API.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.
