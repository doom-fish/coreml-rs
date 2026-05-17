# CoreML.framework coverage audit

Crate: `coreml` 0.2.2  
Framework: `CoreML.framework`  
Headers audited from: `$(xcrun --sdk macosx --show-sdk-path)/System/Library/Frameworks/CoreML.framework/Headers`

Legend:

- ✅ implemented
- 🟡 partial
- ⏭️ skipped

| Header / API surface | Status | Rust / bridge note |
| --- | --- | --- |
| `MLModel.h` | ✅ | `Model` covers compiled-model loading, in-memory asset loading, synchronous prediction, batch prediction, and stateful prediction entry points. |
| `MLModel+MLModelCompilation.h` | ✅ | `ModelCompiler::compile` plus `Model::compile_model` / `Model::compile_and_load`. |
| `MLModelAsset.h` | ✅ | Used for `Model::load_from_specification_data`. |
| `MLModel+MLState.h`, `MLState.h` | ✅ | `Model::new_state`, `Model::predict_with_state`, and `MLState::snapshot_multi_array`. |
| `MLPredictionOptions.h` | ✅ | `PredictionOptions` builder plus bridge round-trip validation. |
| `MLModelConfiguration.h`, `MLOptimizationHints.h`, `MLReshapeFrequencyHint.h`, `MLSpecializationStrategy.h`, `MLParameterKey.h` | ✅ | `ModelConfiguration` exposes compute units, low-precision GPU accumulation, display/function names, optimization hints, and parameter dictionaries. |
| `MLModelDescription.h`, `MLFeatureDescription.h`, `MLParameterDescription.h`, `MLNumericConstraint.h`, `MLModelMetadataKeys.h` | ✅ | `ModelDescription` snapshots cover inputs/outputs/state/training features, metadata, predicted feature names, class labels, and parameter constraints. |
| `MLFeatureValue.h` | ✅ | `Feature` supports int/double/string/multi-array/undefined values and both string- and int-keyed dictionaries. |
| `MLFeatureProvider.h`, `MLDictionaryFeatureProvider.h` | ✅ | `FeatureProvider` / `MLDictionaryFeatureProvider` cover keyed insertion, typed accessors, and generic `Feature` round-trips. |
| `MLBatchProvider.h`, `MLArrayBatchProvider.h` | ✅ | `BatchProvider` / `MLArrayBatchProvider` cover construction, push, count, and indexed access. |
| `MLMultiArray.h` | ✅ | `MultiArray` already covers allocation, shape/stride access, raw slices, and logical indexing helpers. |
| `MLDictionaryConstraint.h` | ✅ | Serialized through `DictionaryConstraint`. |
| `MLStateConstraint.h` | ✅ | Serialized through `StateConstraint`. |
| `MLComputePlan.h` | 🟡 partial | `ComputePlan` returns model-type and graph-count summaries; direct object wrappers are not yet first-class Rust types. |
| `MLComputePlanCost.h`, `MLComputePlanDeviceUsage.h` | �� partial | Summary data is available indirectly through compute-plan inspection, but cost/device-usage wrapper types are deferred. |
| `MLModelStructure.h`, `MLModelStructureNeuralNetwork*.h`, `MLModelStructureProgram*.h`, `MLModelStructurePipeline.h` | �� partial | The bridge walks these objects to build `ComputePlan` summaries, but the full graph is not yet exposed as public Rust structs. |
| `MLAllComputeDevices.h`, `MLCPUComputeDevice.h`, `MLGPUComputeDevice.h`, `MLNeuralEngineComputeDevice.h`, `MLComputeDeviceProtocol.h`, `MLModel+MLComputeDevice.h` | 🟡 partial | Compute-device discovery and preferred-device inspection are not yet wrapped as public Rust APIs. |
| `MLImageConstraint.h`, `MLImageSize.h`, `MLImageSizeConstraint.h`, `MLImageSizeConstraintType.h` | 🟡 partial | `ImageConstraint` exposes width/height/pixel format, but the detailed image-size constraint sub-objects are deferred. |
| `MLMultiArrayConstraint.h`, `MLMultiArrayShapeConstraint.h`, `MLMultiArrayShapeConstraintType.h` | 🟡 partial | `MultiArrayConstraint` exposes shape and scalar type; detailed shape-constraint objects are deferred. |
| `MLSequence.h`, `MLSequenceConstraint.h` | 🟡 partial | Sequence constraint metadata is surfaced through `ModelDescription`, but safe `MLSequence` value wrappers are not yet implemented. |
| `MLFeatureValue+MLImageConversion.h` | ⏭️ skipped | Image-conversion helper category is out of scope until the crate grows image wrapper types beyond raw `CVPixelBuffer` insertion. |
| `MLModelCollection.h`, `MLModelCollectionEntry.h` | ⏭️ skipped | Remote model-collection management is not yet surfaced in the crate. |
| `MLCustomLayer.h`, `MLCustomModel.h` | ✅ | `MLCustomLayerRegistration` / `MLCustomModelRegistration` expose Rust callback authoring, dynamic Objective-C class registration, and headless callback exercise helpers backed by `CustomLayer.swift` / `CustomModel.swift`. |
| `MLWritable.h` | 🟡 partial | Update tasks surface writable models indirectly via captured contexts; direct `writeToURL` wrappers are deferred. |
