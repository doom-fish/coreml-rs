# CoreML.framework coverage audit

Crate: `coreml` 0.4.0  
Framework: `CoreML.framework`  
Headers audited from: Xcode 26.5 / `MacOSX26.5.sdk`

This table is per header. A ✅ row means the header's main types are reachable
from the safe API, not that every method or property is wrapped. Several
surfaces are exposed as serde snapshots (descriptions, compute plans, model
structures) rather than as live object wrappers.

Legend:

- ✅ implemented
- 🟡 partial
- ⏭️ not wrapped

| Header / API surface | Status | Rust / bridge note |
| --- | --- | --- |
| `MLModel.h` | ✅ | `Model` covers compiled-model loading, in-memory asset loading, synchronous and async prediction, batch prediction, and stateful prediction entry points. |
| `MLModel+MLModelCompilation.h` | ✅ | `ModelCompiler::{compile, compile_async, compile_and_load, compile_and_load_async}` plus the `Model::compile_*` helpers. Models from `compile_and_load*` delete their temporary bundle when dropped. |
| `MLModelAsset.h` | 🟡 | Used for `Model::load_from_specification_data`; asset-from-URL and function-name queries are not wrapped. |
| `MLModel+MLState.h`, `MLState.h` | ✅ | `Model::new_state`, `Model::{predict_with_state, predict_with_state_and_options, predict_with_state_async}` (all take `&mut MLState`), and `MLState::{with_multi_array, with_multi_array_mut, snapshot_multi_array}`. The bridge serializes predictions and buffer access per state. |
| `MLPredictionOptions.h` | 🟡 | `PredictionOptions` exposes `uses_cpu_only`; `outputBackings` is not wrapped. |
| `MLModelConfiguration.h`, `MLOptimizationHints.h`, `MLReshapeFrequencyHint.h`, `MLSpecializationStrategy.h`, `MLParameterKey.h` | 🟡 | `ModelConfiguration` exposes compute units, low-precision GPU accumulation, display/function names, optimization hints, and parameter dictionaries; `preferredMetalDevice` is not wrapped. |
| `MLModelDescription.h`, `MLFeatureDescription.h`, `MLParameterDescription.h`, `MLNumericConstraint.h`, `MLModelMetadataKeys.h` | ✅ | `ModelDescription` / `DetailedModelDescription` snapshots; `Model::description` returns an error instead of an empty default when decoding fails. |
| `MLFeatureValue.h` | 🟡 | `Feature` supports int/double/string/multi-array/sequence/undefined values and string- and int-keyed dictionaries; `multi_array_value` returns a read-only view. Pixel-buffer inputs go through `FeatureProvider::insert_cv_pixel_buffer`; image outputs (`imageBufferValue`) are not readable. |
| `MLFeatureValue+MLImageConversion.h` | 🟡 | `Feature::from_image_url(_with_options)` only; the `CGImage` initializers are not wrapped. |
| `MLFeatureProvider.h`, `MLDictionaryFeatureProvider.h` | ✅ | `FeatureProvider` / `MLDictionaryFeatureProvider` cover keyed insertion, typed accessors, and generic `Feature` round-trips. |
| `MLBatchProvider.h`, `MLArrayBatchProvider.h` | ✅ | `BatchProvider` / `MLArrayBatchProvider` cover construction, push, count, and indexed access. |
| `MLMultiArray.h` | 🟡 | `MultiArray` (owned), `MultiArrayRef` (borrowed) and `MultiArrayView` (read-only) cover shape-based allocation, concatenation, transfer, NSNumber-style access, and scoped buffer access through `getBytesWithHandler` / `getMutableBytesWithHandler` (`with_bytes`, `with_slice`, `get`, `set`, `to_vec`, `copy_from_slice`). Element types: Float64, Float32, Float16, Int32, Int8 (macOS 26+); unknown types are reported as `DataType::Unknown` and never read. The custom-stride, data-pointer and pixel-buffer initializers and the `pixelBuffer` accessor are not wrapped. |
| `MLDictionaryConstraint.h`, `MLStateConstraint.h`, `MLSequenceConstraint.h` | ✅ | Serialized through `DictionaryConstraint`, `StateConstraint`, and `SequenceConstraint`. |
| `MLSequence.h` | 🟡 | `MLSequence` covers empty, string and int64 sequences. |
| `MLImageConstraint.h`, `MLImageSize.h`, `MLImageSizeConstraint.h`, `MLImageSizeConstraintType.h` | ✅ | `ImageConstraint` / `DetailedImageConstraint` snapshots, including flexible image sizes. |
| `MLMultiArrayConstraint.h`, `MLMultiArrayShapeConstraint.h`, `MLMultiArrayShapeConstraintType.h` | ✅ | `MultiArrayConstraint` / `DetailedMultiArrayConstraint` snapshots, including shape constraints. |
| `MLComputePlan.h`, `MLComputePlanCost.h`, `MLComputePlanDeviceUsage.h` | 🟡 | `ComputePlan` / `ComputePlanDetails` snapshots with per-operation cost and device usage; no live `MLComputePlan` handle. |
| `MLModelStructure*.h` | 🟡 | `ModelStructure` snapshots of ML Program, neural-network and pipeline structures; no live object handles. |
| `MLAllComputeDevices.h`, `MLCPUComputeDevice.h`, `MLGPUComputeDevice.h`, `MLNeuralEngineComputeDevice.h`, `MLComputeDeviceProtocol.h`, `MLModel+MLComputeDevice.h` | ✅ | `all_compute_devices()` and `Model::available_compute_devices()` return `ComputeDevice` snapshots. |
| `MLUpdateTask.h`, `MLUpdateContext.h`, `MLUpdateProgressHandlers.h`, `MLUpdateProgressEvent.h`, `MLTask.h`, `MLMetricKey.h` | ✅ | `Update::run` returns the updated `Model` plus `UpdateContext` snapshots, calls `UpdateProgressHandlers::on_progress` on the calling thread, and cancels on timeout; there is no live task handle for pause/resume. |
| `MLWritable.h` | ✅ | `Model::write_to_url`. |
| `MLCustomLayer.h`, `MLCustomModel.h` | ✅ | `MLCustomLayerRegistration` / `MLCustomModelRegistration` register Rust callbacks as Objective-C classes; layer callbacks receive borrowed arrays that are never released by Rust. |
| `MLModelCollection.h`, `MLModelCollectionEntry.h` | ⏭️ | Unavailable on macOS in this SDK. |
| `MLTensor` (Swift overlay, macOS 15+) | ⏭️ | Not wrapped. |
| `MLShapedArray`, `MLShapedArraySlice` (Swift overlay) | ⏭️ | Not wrapped. |
