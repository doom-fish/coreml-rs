# Changelog

## [0.2.0] - 2026-05-16

### Added

- `Feature`, `PredictionOptions`, `ComputePlan`, `ModelCompiler`, `Update`, and `MLState` wrappers alongside Apple-style `MLDictionaryFeatureProvider`, `MLArrayBatchProvider`, and `MLModelConfiguration` exports.
- Rich `ModelDescription` snapshots covering state features, training inputs, parameter descriptions, numeric constraints, class labels, and predicted feature metadata.
- Swift bridge split by logical area (`Model`, `ModelDescription`, `Feature`, `Prediction`, `ModelConfiguration`, `ComputePlan`, `ModelCompiler`, `BatchProvider`, `Update`, `MLDictionaryFeatureProvider`, `MLArrayBatchProvider`, `MLState`).
- Twelve area-specific examples and twelve integration-test files covering missing-model error paths, configuration/option round-trips, provider/batch construction, update task wiring, and MLState runtime support.
- `COVERAGE.md` auditing the CoreML SDK headers against the crate surface.

### Changed

- `Model` now supports prediction options, stateful prediction, and delegates compilation helpers through `ModelCompiler`.
- `ModelConfiguration` now serializes through the Swift bridge and exposes function-name, optimization-hint, and parameter support.
- `FeatureProvider` now supports generic `Feature` insertion/retrieval, while `BatchProvider` adds `from_feature_providers` for ergonomic batch construction.

## [0.1.0] - 2026-05-16

### Added

- `Model` wrappers for loading compiled `.mlmodelc` bundles, compiling `.mlmodel` sources, loading in-memory specifications via `MLModelAsset`, and running synchronous single / batch predictions.
- `ModelConfiguration` builder with CoreML compute-unit selection, low-precision GPU accumulation, and display-name support.
- `MultiArray` wrapper with `Float32`, `Float16`, `Int32`, and `Float64` allocation plus raw-slice access, logical indexing, and copy helpers.
- `FeatureProvider` / `BatchProvider` wrappers for dictionaries of tensors, strings, integers, doubles, and `CVPixelBuffer` image inputs.
- `ModelDescription` snapshots covering input/output feature descriptions, tensor constraints, image constraints, metadata, and `is_updatable`.
- SwiftPM bridge (`swift-bridge/`) that links CoreML, Foundation, and CoreVideo into a static library built from `build.rs`.
- Smoke example `examples/01_smoke.rs` that exercises tensor creation, feature-provider round-trips, and clean error handling for a missing model.
