# Changelog

## [0.3.1] - 2026-05-18

- Widen apple-cf version bound to `<0.9` so the 0.8.0 nested-CGRect dep resolves. No source changes.

## [0.3.0] - 2026-05-18

### Added

- New `async` Cargo feature that enables `Model::load_async(path, configuration)` and `Model::predict_async(inputs, options)`, executor-agnostic wrappers over CoreML's asynchronous model loading and single-prediction APIs.
- Integration tests covering asynchronous model loading and prediction against a tiny sentiment-classifier fixture.

## [0.2.2] - 2026-05-17

### Added

- `MLCustomLayerRegistration` / `MLCustomModelRegistration` plus Rust callback traits for bridging CoreML custom layer/model authoring through dynamically registered Objective-C classes.
- Headless examples and integration tests covering custom-layer weight/shape/CPU-evaluation callbacks and custom-model single/batch prediction callbacks.

### Changed

- The Swift bridge now links Metal so `MLCustomLayer` registrations can optionally surface GPU command-buffer encoding hooks.
- `COVERAGE.md` and `COVERAGE_AUDIT.md` now report 92/92 verified public CoreML macOS symbols (100%).

## [0.2.1] - 2026-05-16

### Added

- Compute-device discovery via `all_compute_devices()` and `Model::available_compute_devices()`, plus public `ComputeDevice` / `ComputeDeviceKind` snapshots.
- Rich `ModelStructure` and `ComputePlanDetails` snapshots covering ML Program, neural-network, and pipeline inspection with per-operation cost/device-usage data.
- Detailed `ModelDescription` flexible image-size and multi-array shape constraints, public `MLKey` / `MLModelError` wrappers, `Model::write_to_url`, and `ModelConfiguration::with_ml_key_parameter`.
- `MLSequence` wrappers plus `Feature::from_sequence`, `Feature::sequence_value`, and image feature creation from file URLs with crop/crop-and-scale options.
- `MultiArray::concatenate`, `MultiArray::transfer_to`, and NSNumber-style scalar access helpers.
- New examples and integration tests covering compute-device discovery, model-structure snapshots, detailed model-description decoding, sequence/image features, and the new multi-array helpers.

### Changed

- The Swift bridge now serializes real CoreML compute-plan/model-structure overlays instead of placeholder compute-plan summaries.
- `COVERAGE_AUDIT.md` now verifies 90 of 92 public CoreML top-level symbols on macOS (97.8%), leaving only custom layer/model authoring protocols as open gaps.

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
