# Changelog

All notable changes to `coreml` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-24

### Security

- `MultiArray::data_type` no longer reports unknown element types as `Float32`. An `Int8` (macOS 26) model output read as `f32` spanned four times its storage, an out-of-bounds read and write from safe code.
- `FeatureProvider::get_multi_array` and `Feature::multi_array_value` no longer return a new owning wrapper around the same `MLMultiArray` on every call, which let safe code hold `&mut` and `&` to one buffer.
- A panicking custom-layer callback no longer releases CoreML's lent arrays a second time (use-after-free); this completes the 0.3.5 fix.
- The compute-plan and model-structure bridges no longer read the caller's path inside an escaping Swift task after the call has returned.

### Fixed

- Multi-array access runs inside `getBytesWithHandler` / `getMutableBytesWithHandler` instead of long-lived slices over the deprecated `dataPointer`; every offset is bounds-checked and follows the strides the handler reports.
- `MLState::snapshot_multi_array` copies strided state buffers element by element instead of `count` elements linearly.
- `Model::new_state` no longer leaks every `MLState` and its buffers.
- Stateful predictions on one `MLState` are serialized, and buffer access waits for predictions still in flight, including those of dropped futures. Async predictions wait for the state as queued continuations instead of each blocking a GCD worker thread, so a backlog of timed-out or dropped predictions cannot starve the dispatch pool that the running prediction needs.
- Custom layer and custom model instances are locked instead of being cast to `&mut` from concurrent CoreML threads.
- `Update::run` returns the updated model, delivers progress callbacks, and no longer cancels training after a fixed 60 s.
- The blocking wrappers over CoreML's async APIs no longer fail after a fixed 60 s while the work keeps running; a configured timeout cancels the work and reports `TimedOut`.
- Dropping an async future cancels its CoreML task, and async predictions work on a snapshot of the input provider.
- `FeatureProvider::insert_*` report an interior NUL as an error instead of panicking.
- `MultiArray::transfer_to` and `concatenate` validate shapes and element types before the Objective-C call.
- `copy_from_slice` takes the logical element count; the old `copy_from_*_slice` required the padded storage length.
- `Model::description` and `detailed_description` report serialization and decode errors instead of returning an empty description.
- Temporary `.mlmodelc` bundles are removed: `compile_and_load(_async)` models delete theirs when dropped, and bundles that arrive after a timeout or for a dropped compile future are deleted.
- Paths that are not valid UTF-8 are rejected instead of being converted lossily.
- New multi-arrays are zero-filled.
- The custom-layer and custom-model integration tests run again.
- Bridge JSON (update progress and completion contexts, dictionary features, custom-layer and custom-model parameters, configuration parameters and description metadata) carries NaN and infinite numbers as the strings `"NaN"`, `"Infinity"` and `"-Infinity"`. A NaN training loss used to collapse the update context to `{}`, so `Update::run` failed and dropped the trained model, and `Feature::string_dictionary_value` and `int64_dictionary_value` returned an empty dictionary when one value was NaN.
- `MLCustomLayerHandle::output_shapes_for_input_shapes` reports input dimensions above `Int.max` as `InvalidArgument` instead of passing the layer an empty shape list.
- A panic in the `Drop` of a custom layer or custom model no longer aborts the process when CoreML releases the instance.
- `build.rs` no longer adds the toolchain's Swift 5.5 back-deployment directory (`usr/lib/swift-5.5/macosx`) to the rpath. The path points into Xcode, so it never made the back-deployment concurrency library available on other machines.

### Changed

- **Breaking:** `DataType` gains `Int8` and `Unknown(i64)`, and `MultiArrayScalar` gains `Int8`; both are `#[non_exhaustive]`.
- **Breaking:** `MultiArray` dereferences to `MultiArrayRef`. The per-type `as_*_slice(_mut)`, `get_*`, `set_*` and `copy_from_*_slice` methods are replaced by generic `with_slice(_mut)`, `with_bytes(_mut)`, `get`, `set`, `to_vec` and `copy_from_slice` over `MultiArrayElement` (`f64`, `f32`, `f16`, `i32`, `i8`); `get` returns a `Result`.
- **Breaking:** `MultiArray::concatenate` takes `&[impl AsRef<MultiArrayRef>]` and `transfer_to` takes `&mut MultiArrayRef`.
- **Breaking:** `FeatureProvider::get_multi_array` and `Feature::multi_array_value` return `MultiArrayView<'_>`.
- **Breaking:** `MLCustomLayer::evaluate_on_cpu` and `MLCustomLayerHandle::evaluate_on_cpu` take `&[&MultiArrayRef]` and `&mut [&mut MultiArrayRef]`.
- **Breaking:** `Model::predict_with_state`, `predict_with_state_and_options` and `predict_with_state_async` take `&mut MLState`.
- **Breaking:** `Update::run` takes the handlers by value plus a `timeout: Option<Duration>` and returns `UpdateOutcome { model, result }`; `UpdateProgressHandlers` has a lifetime and no longer implements `Clone`, `PartialEq` or serde.
- **Breaking:** `FeatureProvider::insert_*` and `Model::description` / `detailed_description` return `Result`.
- **Breaking:** `BatchProvider::try_push` is removed. `push` cannot fail and no longer hides an `expect` on the bridge status.
- **Breaking:** raw FFI: multi-array data types cross as `NSInteger`; the async exports return a cancellable task handle; the blocking exports take a timeout; `cm_update_run` is replaced by `cm_update_start` / `cm_update_cancel` and `cm_state_snapshot_multi_array` by `cm_state_with_multi_array`.
- `doom-fish-utils` is a regular dependency. Requires `apple-cf` 0.11 and `doom-fish-utils` 0.4.1; `rust-version` is 1.82.

### Added

- `MultiArrayRef`, `MultiArrayView`, `MultiArrayElement`, `DataType::element_size` and `MultiArrayRef::copy_to_owned`.
- `MLState::with_multi_array` and `with_multi_array_mut`.
- `UpdateProgressHandlers::on_progress` and `UpdateOutcome`.
- `set_blocking_timeout` and `blocking_timeout`.

### Removed

- `FeatureProvider::try_insert_*`; use `insert_*`.
- `BatchProvider::try_push`; use `push`.
- The raw `cm_multi_array_data_pointer` export.

## [0.3.5] - 2026-06-06

- Size raw multi-array slices to the strided storage extent, and stop double-releasing feature providers and arrays lent to custom-model and custom-layer callbacks.

## [0.3.4] - 2026-05-20

- Phase 32 completeness + async sweep.
- Added async CoreML model compilation helpers (`ModelCompiler::compile_async`, `Model::compile_model_async`, `Model::compile_and_load_async`) and async stateful prediction via `Model::predict_with_state_async`.
- Refreshed the coverage docs against `MacOSX26.5.sdk`.

## [0.3.3] - 2026-05-20

- Widen `doom-fish-utils` dependency bound to `<0.4` so the 0.3.x SPSC-ring release resolves cleanly. No source changes.

## [0.3.2] - 2026-05-18

- Widen apple-cf version bound to `<0.10` so 0.9.x resolves.

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
