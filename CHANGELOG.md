# Changelog

## [0.1.0] - 2026-05-16

### Added

- `Model` wrappers for loading compiled `.mlmodelc` bundles, compiling `.mlmodel` sources, loading in-memory specifications via `MLModelAsset`, and running synchronous single / batch predictions.
- `ModelConfiguration` builder with CoreML compute-unit selection, low-precision GPU accumulation, and display-name support.
- `MultiArray` wrapper with `Float32`, `Float16`, `Int32`, and `Float64` allocation plus raw-slice access, logical indexing, and copy helpers.
- `FeatureProvider` / `BatchProvider` wrappers for dictionaries of tensors, strings, integers, doubles, and `CVPixelBuffer` image inputs.
- `ModelDescription` snapshots covering input/output feature descriptions, tensor constraints, image constraints, metadata, and `is_updatable`.
- SwiftPM bridge (`swift-bridge/`) that links CoreML, Foundation, and CoreVideo into a static library built from `build.rs`.
- Smoke example `examples/01_smoke.rs` that exercises tensor creation, feature-provider round-trips, and clean error handling for a missing model.
