# coreml coverage audit (vs MacOSX26.2.sdk)

SDK_PUBLIC_SYMBOLS: 92
VERIFIED: 53
GAPS: 39
EXEMPT: 0
COVERAGE_PCT: 57.6%

This audit counts top-level Objective-C CoreML symbols (interfaces, protocols, enums, exported constants, and top-level C functions), not every individual method/property. Stringly-typed/raw-JSON access counts as **VERIFIED** when the crate’s public API can still reach the underlying framework surface (for example metadata keys, parameter keys, and metric keys).

## Not counted (macOS unavailable in MacOSX26.2.sdk)

| Symbol | Kind | Header | Notes |
| --- | --- | --- | --- |
| MLModelCollection | interface | MLModelCollection.h | Excluded from counts: `MODELCOLLECTION_SUNSET(...)` resolves to `API_UNAVAILABLE(macos)` in this SDK. |
| MLModelCollectionEntry | interface | MLModelCollectionEntry.h | Excluded from counts: `MODELCOLLECTION_SUNSET(...)` resolves to `API_UNAVAILABLE(macos)` in this SDK. |
| MLModelCollectionDidChangeNotification | constant | MLModelCollection.h | Excluded from counts: `MODELCOLLECTION_SUNSET(...)` resolves to `API_UNAVAILABLE(macos)` in this SDK. |

## 🟢 VERIFIED
| Symbol | Kind | Header | Wrapped by |
| --- | --- | --- | --- |
| MLArrayBatchProvider | interface | MLArrayBatchProvider.h | `MLArrayBatchProvider` alias (`BatchProvider`) via `src/ml_array_batch_provider.rs`; backed by `BatchProvider` in `src/feature_provider/mod.rs`. |
| MLBatchProvider | protocol | MLBatchProvider.h | `BatchProvider` in `src/feature_provider/mod.rs`. |
| MLComputePlan | interface | MLComputePlan.h | `ComputePlan` / `ComputePlan::load_from_url` in `src/compute_plan.rs` (the Swift bridge currently serializes only placeholder `unknown/0` summary fields in `swift-bridge/Sources/CoreMLBridge/ComputePlan.swift`). |
| MLDictionaryConstraint | interface | MLDictionaryConstraint.h | `DictionaryConstraint` snapshot in `src/model_description.rs`. |
| MLDictionaryFeatureProvider | interface | MLDictionaryFeatureProvider.h | `MLDictionaryFeatureProvider` alias (`FeatureProvider`) via `src/ml_dictionary_feature_provider.rs`. |
| MLFeatureDescription | interface | MLFeatureDescription.h | `FeatureDescription` snapshot in `src/model_description.rs`. |
| MLFeatureDescription (MLFeatureValueConstraints) | interface | MLFeatureDescription.h | `FeatureDescription` constraint fields (`multi_array_constraint`, `image_constraint`, `dictionary_constraint`, `sequence_constraint`, `state_constraint`) in `src/model_description.rs`. |
| MLFeatureProvider | protocol | MLFeatureProvider.h | `FeatureProvider` in `src/feature_provider/mod.rs`. |
| MLFeatureType | enum | MLFeatureType.h | `FeatureType` enum in `src/feature.rs`. |
| MLFeatureValue | interface | MLFeatureValue.h | `Feature` in `src/feature.rs` (int/double/string/multi-array/undefined/dictionary; no image/sequence factories). |
| MLImageConstraint | interface | MLImageConstraint.h | `ImageConstraint` snapshot in `src/model_description.rs` (fixed size + pixel format only). |
| MLMetricKey | interface | MLMetricKey.h | `UpdateContext::metrics: BTreeMap<String, Value>` in `src/update.rs`; the Swift bridge stringifies `MLMetricKey` names in `swift-bridge/Sources/CoreMLBridge/Update.swift`. |
| MLModel (MLModelCompilation) | interface | MLModel+MLModelCompilation.h | `ModelCompiler::compile`, `Model::compile_model`, and `Model::compile_and_load` in `src/model_compiler.rs` and `src/model/mod.rs`. |
| MLModel (MLState) | interface | MLModel+MLState.h | `Model::new_state`, `predict_with_state`, and `predict_with_state_and_options` in `src/model/mod.rs`. |
| MLModel | interface | MLModel.h | `Model` in `src/model/mod.rs` (`load_from_url`, `predict*`, `predict_batch*`, `description`). |
| MLModelAsset | interface | MLModelAsset.h | `Model::load_from_specification_data` in `src/model/mod.rs`; the Swift bridge constructs `MLModelAsset(specification:)` in `swift-bridge/Sources/CoreMLBridge/Model.swift`. |
| MLComputeUnits | enum | MLModelConfiguration.h | `ComputeUnits` in `src/configuration/mod.rs`. |
| MLModelConfiguration | interface | MLModelConfiguration.h | `ModelConfiguration` builder in `src/configuration/mod.rs`. |
| MLModelConfiguration (MLGPUConfigurationOptions) | interface | MLModelConfiguration.h | `ModelConfiguration::with_allow_low_precision_accumulation_on_gpu` in `src/configuration/mod.rs` (no `preferredMetalDevice` wrapper). |
| MLModelConfiguration (MLModelParameterAdditions) | interface | MLModelConfiguration.h | `ModelConfiguration::with_parameter` / `parameters()` in `src/configuration/mod.rs` with Swift key mapping in `swift-bridge/Sources/CoreMLBridge/Core.swift`. |
| MLModelConfiguration (MultiFunctions) | interface | MLModelConfiguration.h | `ModelConfiguration::with_function_name` / `function_name()` in `src/configuration/mod.rs`. |
| MLModelDescription | interface | MLModelDescription.h | `ModelDescription` snapshot in `src/model_description.rs`. |
| MLModelDescription (MLUpdateAdditions) | interface | MLModelDescription.h | `ModelDescription::training_inputs` and `is_updatable` in `src/model_description.rs`. |
| MLModelDescription (MLParameters) | interface | MLModelDescription.h | `ModelDescription::parameter_descriptions` / `ParameterDescription` in `src/model_description.rs`. |
| MLModelDescriptionKey | constant | MLModelMetadataKeys.h | `ModelDescription::metadata` in `src/model_description.rs` exposes metadata as raw string keys. |
| MLModelVersionStringKey | constant | MLModelMetadataKeys.h | `ModelDescription::metadata` in `src/model_description.rs` exposes metadata as raw string keys. |
| MLModelAuthorKey | constant | MLModelMetadataKeys.h | `ModelDescription::metadata` in `src/model_description.rs` exposes metadata as raw string keys. |
| MLModelLicenseKey | constant | MLModelMetadataKeys.h | `ModelDescription::metadata` in `src/model_description.rs` exposes metadata as raw string keys. |
| MLModelCreatorDefinedKey | constant | MLModelMetadataKeys.h | `ModelDescription::metadata` in `src/model_description.rs` exposes metadata as raw string keys. |
| MLMultiArrayDataType | enum | MLMultiArray.h | `DataType` in `src/multi_array/mod.rs`. |
| MLMultiArray | interface | MLMultiArray.h | `MultiArray` in `src/multi_array/mod.rs`. |
| MLMultiArray (Creation) | interface | MLMultiArray.h | `MultiArray::new*` constructors in `src/multi_array/mod.rs` (shape-based allocation only; no custom-stride/data-pointer/pixel-buffer initializers). |
| MLMultiArray (ScopedBufferAccess) | interface | MLMultiArray.h | `MultiArray::as_*_slice`, `as_*_slice_mut`, and index helpers in `src/multi_array/mod.rs`. |
| MLMultiArrayConstraint | interface | MLMultiArrayConstraint.h | `MultiArrayConstraint` snapshot in `src/model_description.rs`. |
| MLNumericConstraint | interface | MLNumericConstraint.h | `NumericConstraint` snapshot in `src/model_description.rs`. |
| MLOptimizationHints | interface | MLOptimizationHints.h | `OptimizationHints` in `src/configuration/mod.rs`. |
| MLParameterDescription | interface | MLParameterDescription.h | `ParameterDescription` snapshot in `src/model_description.rs`. |
| MLParameterKey | interface | MLParameterKey.h | `ModelConfiguration::with_parameter` accepts CoreML parameter names; the Swift bridge maps names to `MLParameterKey` in `swift-bridge/Sources/CoreMLBridge/Core.swift`. |
| MLParameterKey (MLLinkedModelParameters) | interface | MLParameterKey.h | `ModelConfiguration::with_parameter("linked_model_file_name"|"linked_model_search_path", …)` reaches this surface via the Swift string-to-key mapping. |
| MLParameterKey (MLNeuralNetworkParameters) | interface | MLParameterKey.h | `ModelConfiguration::with_parameter("weights"|"biases", …)` reaches this surface via the Swift string-to-key mapping. |
| MLParameterKey (MLScopedParameters) | interface | MLParameterKey.h | `ModelConfiguration::with_parameter("name:scope", …)` uses the bridge’s scoped-key parsing in `swift-bridge/Sources/CoreMLBridge/Core.swift`. |
| MLPredictionOptions | interface | MLPredictionOptions.h | `PredictionOptions` in `src/prediction.rs` (only `uses_cpu_only`; no `outputBackings`). |
| MLReshapeFrequencyHint | enum | MLReshapeFrequencyHint.h | `ReshapeFrequencyHint` in `src/configuration/mod.rs`. |
| MLSequenceConstraint | interface | MLSequenceConstraint.h | `SequenceConstraint` snapshot in `src/model_description.rs`. |
| MLSpecializationStrategy | enum | MLSpecializationStrategy.h | `SpecializationStrategy` in `src/configuration/mod.rs`. |
| MLState | interface | MLState.h | `MLState` in `src/ml_state.rs` (`runtime_supported`, `snapshot_multi_array`). |
| MLStateConstraint | interface | MLStateConstraint.h | `StateConstraint` snapshot in `src/model_description.rs`. |
| MLTaskState | enum | MLTask.h | `UpdateTaskState` in `src/update.rs`. |
| MLTask | interface | MLTask.h | `UpdateContext::task_identifier` and `UpdateContext::state` in `src/update.rs` surface `MLTask` state snapshots (no direct task handle). |
| MLUpdateContext | interface | MLUpdateContext.h | `UpdateContext` in `src/update.rs`. |
| MLUpdateProgressEvent | enum | MLUpdateProgressEvent.h | `UpdateEvent` in `src/update.rs`. |
| MLUpdateProgressHandlers | interface | MLUpdateProgressHandlers.h | `UpdateProgressHandlers` in `src/update.rs`. |
| MLUpdateTask | interface | MLUpdateTask.h | `Update::run` in `src/update.rs` drives `MLUpdateTask` synchronously; the Swift bridge creates the task in `swift-bridge/Sources/CoreMLBridge/Update.swift`. |

## 🔴 GAPS
| Symbol | Kind | Header | Notes |
| --- | --- | --- | --- |
| MLAllComputeDevices | function | MLAllComputeDevices.h | No public Rust wrapper for the top-level compute-device discovery function. |
| MLCPUComputeDevice | interface | MLCPUComputeDevice.h | No public compute-device object wrapper. |
| MLComputeDeviceProtocol | protocol | MLComputeDeviceProtocol.h | No protocol/object model for CoreML compute devices. |
| MLComputePlanCost | interface | MLComputePlanCost.h | No public per-operation cost wrapper. |
| MLComputePlanDeviceUsage | interface | MLComputePlanDeviceUsage.h | No public per-operation device-usage wrapper. |
| MLCustomLayer | protocol | MLCustomLayer.h | Custom layer authoring callbacks are not bridged. |
| MLCustomModel | protocol | MLCustomModel.h | Custom model authoring callbacks are not bridged. |
| MLFeatureValueImageOptionCropRect | constant | MLFeatureValue+MLImageConversion.h | Image-conversion option constants are not surfaced. |
| MLFeatureValueImageOptionCropAndScale | constant | MLFeatureValue+MLImageConversion.h | Image-conversion option constants are not surfaced. |
| MLFeatureValue (MLImageConversion) | interface | MLFeatureValue+MLImageConversion.h | No image-conversion factories (`featureValueWithImageAtURL:` / `featureValueWithCGImage:`) are bridged. |
| MLGPUComputeDevice | interface | MLGPUComputeDevice.h | No public compute-device object wrapper. |
| MLImageSize | interface | MLImageSize.h | No public wrapper for enumerated/flexible image sizes. |
| MLImageSizeConstraint | interface | MLImageSizeConstraint.h | Flexible image-size ranges are not modelled. |
| MLImageSizeConstraintType | enum | MLImageSizeConstraintType.h | No enum wrapper for image-size constraint kinds. |
| MLKey | interface | MLKey.h | No public base key wrapper. |
| MLModel (MLComputeDevice) | interface | MLModel+MLComputeDevice.h | No runtime API for available/preferred compute devices on `Model`. |
| MLModelErrorDomain | constant | MLModelError.h | The framework error-domain constant is not re-exposed. |
| MLModelError | enum | MLModelError.h | `CoreMLError` wraps custom bridge statuses, not `MLModelError` codes. |
| MLModelStructure | interface | MLModelStructure.h | The crate does not expose `MLModelStructure`; compute-plan graph inspection is absent. |
| MLModelStructureNeuralNetwork | interface | MLModelStructureNeuralNetwork.h | No public wrapper for neural-network structure inspection. |
| MLModelStructureNeuralNetworkLayer | interface | MLModelStructureNeuralNetworkLayer.h | No public wrapper for layer inspection. |
| MLModelStructurePipeline | interface | MLModelStructurePipeline.h | No public wrapper for pipeline-structure inspection. |
| MLModelStructureProgram | interface | MLModelStructureProgram.h | No public wrapper for ML Program structure inspection. |
| MLModelStructureProgramArgument | interface | MLModelStructureProgramArgument.h | No public wrapper for program arguments. |
| MLModelStructureProgramBinding | interface | MLModelStructureProgramBinding.h | No public wrapper for program bindings. |
| MLModelStructureProgramBlock | interface | MLModelStructureProgramBlock.h | No public wrapper for program blocks. |
| MLModelStructureProgramFunction | interface | MLModelStructureProgramFunction.h | No public wrapper for program functions. |
| MLModelStructureProgramNamedValueType | interface | MLModelStructureProgramNamedValueType.h | No public wrapper for named value types. |
| MLModelStructureProgramOperation | interface | MLModelStructureProgramOperation.h | No public wrapper for program operations. |
| MLModelStructureProgramValue | interface | MLModelStructureProgramValue.h | No public wrapper for program values. |
| MLModelStructureProgramValueType | interface | MLModelStructureProgramValueType.h | No public wrapper for program value types. |
| MLMultiArray (Concatenating) | interface | MLMultiArray.h | No wrapper for `multiArrayByConcatenatingMultiArrays:alongAxis:dataType:`. |
| MLMultiArray (NSNumberDataAccess) | interface | MLMultiArray.h | No NSNumber/subscript accessors are bridged. |
| MLMultiArray (Transferring) | interface | MLMultiArray.h | No wrapper for `transferToMultiArray:`. |
| MLMultiArrayShapeConstraint | interface | MLMultiArrayShapeConstraint.h | Detailed flexible-shape constraint objects are not surfaced. |
| MLMultiArrayShapeConstraintType | enum | MLMultiArrayShapeConstraintType.h | No enum wrapper for multi-array shape-constraint kinds. |
| MLNeuralEngineComputeDevice | interface | MLNeuralEngineComputeDevice.h | No public compute-device object wrapper. |
| MLSequence | interface | MLSequence.h | `FeatureType::Sequence` exists, but there is no public sequence-value wrapper or constructor. |
| MLWritable | protocol | MLWritable.h | No public wrapper for `writeToURL:error:`. |

## ⏭️ EXEMPT
| Symbol | Kind | Header | Reason | SDK attribute |
| --- | --- | --- | --- | --- |
| _(none)_ | - | - | No deprecated top-level macOS symbols were part of the 92-symbol audited set. | - |
