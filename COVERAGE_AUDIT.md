# coreml coverage audit (vs MacOSX26.2.sdk)

SDK_PUBLIC_SYMBOLS: 92
VERIFIED: 90
GAPS: 2
EXEMPT: 0
COVERAGE_PCT: 97.8%

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
| MLComputePlan | interface | MLComputePlan.h | `ComputePlan` / `ComputePlanDetails` in `src/compute_plan.rs`, backed by real overlay serialization in `swift-bridge/Sources/CoreMLBridge/ComputePlan.swift`. |
| MLDictionaryConstraint | interface | MLDictionaryConstraint.h | `DictionaryConstraint` snapshot in `src/model_description.rs`. |
| MLDictionaryFeatureProvider | interface | MLDictionaryFeatureProvider.h | `MLDictionaryFeatureProvider` alias (`FeatureProvider`) via `src/ml_dictionary_feature_provider.rs`. |
| MLFeatureDescription | interface | MLFeatureDescription.h | `FeatureDescription` snapshot in `src/model_description.rs`. |
| MLFeatureDescription (MLFeatureValueConstraints) | interface | MLFeatureDescription.h | `FeatureDescription` constraint fields (`multi_array_constraint`, `image_constraint`, `dictionary_constraint`, `sequence_constraint`, `state_constraint`) in `src/model_description.rs`. |
| MLFeatureProvider | protocol | MLFeatureProvider.h | `FeatureProvider` in `src/feature_provider/mod.rs`. |
| MLFeatureType | enum | MLFeatureType.h | `FeatureType` enum in `src/feature.rs`. |
| MLFeatureValue | interface | MLFeatureValue.h | `Feature` in `src/feature.rs` (int/double/string/multi-array/image/sequence/undefined/dictionary). |
| MLImageConstraint | interface | MLImageConstraint.h | `ImageConstraint` / `DetailedImageConstraint` snapshots in `src/model_description.rs` (fixed size + flexible image-size metadata). |
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
| MLAllComputeDevices | function | MLAllComputeDevices.h | `all_compute_devices()` in `src/compute_device.rs`; bridged by `cm_all_compute_devices_json` in `swift-bridge/Sources/CoreMLBridge/ComputeDevice.swift`. |
| MLCPUComputeDevice | interface | MLCPUComputeDevice.h | `ComputeDevice` / `ComputeDeviceKind::Cpu` in `src/compute_device.rs`. |
| MLComputeDeviceProtocol | protocol | MLComputeDeviceProtocol.h | `ComputeDevice` snapshots in `src/compute_device.rs` serialize CoreML compute devices discovered by the Swift bridge. |
| MLComputePlanCost | interface | MLComputePlanCost.h | `ComputePlanCost` in `src/compute_plan.rs`. |
| MLComputePlanDeviceUsage | interface | MLComputePlanDeviceUsage.h | `ComputePlanDeviceUsage` in `src/compute_plan.rs`. |
| MLFeatureValueImageOptionCropRect | constant | MLFeatureValue+MLImageConversion.h | `ImageFeatureOptions::crop_rect` in `src/feature.rs`; mapped in `swift-bridge/Sources/CoreMLBridge/Feature.swift`. |
| MLFeatureValueImageOptionCropAndScale | constant | MLFeatureValue+MLImageConversion.h | `ImageCropAndScale` / `ImageFeatureOptions::crop_and_scale` in `src/feature.rs`; mapped in `swift-bridge/Sources/CoreMLBridge/Feature.swift`. |
| MLFeatureValue (MLImageConversion) | interface | MLFeatureValue+MLImageConversion.h | `Feature::from_image_url` / `Feature::from_image_url_with_options` in `src/feature.rs`. |
| MLGPUComputeDevice | interface | MLGPUComputeDevice.h | `ComputeDevice` / `ComputeDeviceKind::Gpu` in `src/compute_device.rs`. |
| MLImageSize | interface | MLImageSize.h | `ImageSize` in `src/model_description.rs`. |
| MLImageSizeConstraint | interface | MLImageSizeConstraint.h | `ImageSizeConstraint` in `src/model_description.rs`. |
| MLImageSizeConstraintType | enum | MLImageSizeConstraintType.h | `ImageSizeConstraintType` in `src/model_description.rs`. |
| MLKey | interface | MLKey.h | `MLKey` in `src/ml_key.rs`, plus `ParameterDescription::ml_key` and `ModelConfiguration::with_ml_key_parameter`. |
| MLModel (MLComputeDevice) | interface | MLModel+MLComputeDevice.h | `Model::available_compute_devices()` in `src/model/mod.rs`. |
| MLModelErrorDomain | constant | MLModelError.h | `ML_MODEL_ERROR_DOMAIN` in `src/model_error.rs`. |
| MLModelError | enum | MLModelError.h | `MLModelError` in `src/model_error.rs`. |
| MLModelStructure | interface | MLModelStructure.h | `ModelStructure` in `src/model_structure.rs`; also surfaced through `ComputePlanDetails`. |
| MLModelStructureNeuralNetwork | interface | MLModelStructureNeuralNetwork.h | `NeuralNetwork` snapshots in `src/model_structure.rs`. |
| MLModelStructureNeuralNetworkLayer | interface | MLModelStructureNeuralNetworkLayer.h | `NeuralNetworkLayer` in `src/model_structure.rs`. |
| MLModelStructurePipeline | interface | MLModelStructurePipeline.h | `Pipeline` in `src/model_structure.rs`. |
| MLModelStructureProgram | interface | MLModelStructureProgram.h | `Program` in `src/model_structure.rs`. |
| MLModelStructureProgramArgument | interface | MLModelStructureProgramArgument.h | `ProgramArgument` in `src/model_structure.rs`. |
| MLModelStructureProgramBinding | interface | MLModelStructureProgramBinding.h | `ProgramBinding` in `src/model_structure.rs`. |
| MLModelStructureProgramBlock | interface | MLModelStructureProgramBlock.h | `ProgramBlock` in `src/model_structure.rs`. |
| MLModelStructureProgramFunction | interface | MLModelStructureProgramFunction.h | `ProgramFunction` in `src/model_structure.rs`. |
| MLModelStructureProgramNamedValueType | interface | MLModelStructureProgramNamedValueType.h | `ProgramNamedValueType` in `src/model_structure.rs`. |
| MLModelStructureProgramOperation | interface | MLModelStructureProgramOperation.h | `ProgramOperation` in `src/model_structure.rs`; referenced by `ComputePlanProgramOperationPlan`. |
| MLModelStructureProgramValue | interface | MLModelStructureProgramValue.h | `ProgramValue` in `src/model_structure.rs`. |
| MLModelStructureProgramValueType | interface | MLModelStructureProgramValueType.h | `ProgramValueType` in `src/model_structure.rs`. |
| MLMultiArray (Concatenating) | interface | MLMultiArray.h | `MultiArray::concatenate` in `src/multi_array/mod.rs`. |
| MLMultiArray (NSNumberDataAccess) | interface | MLMultiArray.h | `MultiArray::number_at_linear_index`, `set_number_at_linear_index`, `number_at_indices`, and `set_number_at_indices` in `src/multi_array/mod.rs`. |
| MLMultiArray (Transferring) | interface | MLMultiArray.h | `MultiArray::transfer_to` in `src/multi_array/mod.rs`. |
| MLMultiArrayShapeConstraint | interface | MLMultiArrayShapeConstraint.h | `MultiArrayShapeConstraint` in `src/model_description.rs`. |
| MLMultiArrayShapeConstraintType | enum | MLMultiArrayShapeConstraintType.h | `MultiArrayShapeConstraintType` in `src/model_description.rs`. |
| MLNeuralEngineComputeDevice | interface | MLNeuralEngineComputeDevice.h | `ComputeDevice` / `ComputeDeviceKind::NeuralEngine` in `src/compute_device.rs`. |
| MLSequence | interface | MLSequence.h | `MLSequence` in `src/ml_sequence.rs`, plus `Feature::from_sequence` / `Feature::sequence_value` in `src/feature.rs`. |
| MLWritable | protocol | MLWritable.h | `Model::write_to_url` in `src/model/mod.rs`; bridged by `cm_model_write_to_url` in `swift-bridge/Sources/CoreMLBridge/Model.swift`. |

## 🔴 GAPS
| Symbol | Kind | Header | Notes |
| --- | --- | --- | --- |
| MLCustomLayer | protocol | MLCustomLayer.h | Custom layer authoring callbacks are not bridged. |
| MLCustomModel | protocol | MLCustomModel.h | Custom model authoring callbacks are not bridged. |

## ⏭️ EXEMPT
| Symbol | Kind | Header | Reason | SDK attribute |
| --- | --- | --- | --- | --- |
| _(none)_ | - | - | No deprecated top-level macOS symbols were part of the 92-symbol audited set. | - |
