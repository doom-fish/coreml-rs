import CoreML
import Foundation

func cm_dimension_range_object(_ range: NSRange) -> [String: Any] {
  [
    "lower": range.lowerBound,
    "upper": range.upperBound,
  ]
}

func cm_image_size_object(_ imageSize: MLImageSize) -> [String: Any] {
  [
    "pixels_wide": imageSize.pixelsWide,
    "pixels_high": imageSize.pixelsHigh,
  ]
}

func cm_image_size_constraint_type_name(_ type: MLImageSizeConstraintType) -> String {
  switch type.rawValue {
  case 2:
    return "enumerated"
  case 3:
    return "range"
  default:
    return "unspecified"
  }
}

func cm_image_size_constraint_object(_ constraint: MLImageSizeConstraint) -> [String: Any] {
  [
    "constraint_type": cm_image_size_constraint_type_name(constraint.type),
    "pixels_wide_range": cm_dimension_range_object(constraint.pixelsWideRange),
    "pixels_high_range": cm_dimension_range_object(constraint.pixelsHighRange),
    "enumerated_image_sizes": constraint.enumeratedImageSizes.map(cm_image_size_object),
  ]
}

func cm_multi_array_shape_constraint_type_name(_ type: MLMultiArrayShapeConstraintType) -> String {
  switch type.rawValue {
  case 2:
    return "enumerated"
  case 3:
    return "range"
  default:
    return "unspecified"
  }
}

func cm_multi_array_shape_constraint_object(_ constraint: MLMultiArrayShapeConstraint) -> [String:
  Any]
{
  [
    "constraint_type": cm_multi_array_shape_constraint_type_name(constraint.type),
    "size_ranges": constraint.sizeRangeForDimension.map {
      cm_dimension_range_object($0.rangeValue)
    },
    "enumerated_shapes": constraint.enumeratedShapes.map { $0.map(\.intValue) },
  ]
}

func cm_feature_description_object(_ description: MLFeatureDescription) -> [String: Any] {
  var object: [String: Any] = [
    "name": description.name,
    "feature_type": cm_feature_type_name(description.type),
    "optional": description.isOptional,
  ]

  if let multiArrayConstraint = description.multiArrayConstraint {
    var multiArrayObject: [String: Any] = [
      "shape": multiArrayConstraint.shape.map(\.intValue),
      "data_type": cm_multi_array_data_type_name(multiArrayConstraint.dataType),
    ]
    multiArrayObject["shape_constraint"] = cm_multi_array_shape_constraint_object(
      multiArrayConstraint.shapeConstraint
    )
    object["multi_array_constraint"] = multiArrayObject
  }

  if let imageConstraint = description.imageConstraint {
    var imageObject: [String: Any] = [
      "pixels_wide": imageConstraint.pixelsWide,
      "pixels_high": imageConstraint.pixelsHigh,
      "pixel_format_type": UInt32(imageConstraint.pixelFormatType),
    ]
    imageObject["size_constraint"] = cm_image_size_constraint_object(imageConstraint.sizeConstraint)
    object["image_constraint"] = imageObject
  }

  if let dictionaryConstraint = description.dictionaryConstraint {
    object["dictionary_constraint"] = [
      "key_type": cm_feature_type_name(dictionaryConstraint.keyType)
    ]
  }

  if let sequenceConstraint = description.sequenceConstraint {
    object["sequence_constraint"] = [
      "value_type": cm_feature_type_name(sequenceConstraint.valueDescription.type),
      "count_range_lower": sequenceConstraint.countRange.lowerBound,
      "count_range_upper": sequenceConstraint.countRange.upperBound,
    ]
  }

  #if COREML_HAS_MACOS15_SDK
    if #available(macOS 15.0, *), let stateConstraint = description.stateConstraint {
      object["state_constraint"] = [
        "buffer_shape": stateConstraint.bufferShape,
        "data_type": cm_multi_array_data_type_name(stateConstraint.dataType),
      ]
    }
  #endif

  return object
}

func cm_numeric_constraint_object(_ constraint: MLNumericConstraint) -> [String: Any] {
  [
    "min": constraint.minNumber.doubleValue,
    "max": constraint.maxNumber.doubleValue,
    "enumerated": (constraint.enumeratedNumbers ?? []).map(\.doubleValue).sorted(),
  ]
}

func cm_parameter_description_object(_ description: MLParameterDescription) -> [String: Any] {
  var mlKey: [String: Any] = [
    "name": description.key.name
  ]
  if let scope = description.key.scope {
    mlKey["scope"] = scope
  }
  var object: [String: Any] = [
    "key": description.key.name,
    "default_value": cm_json_safe(description.defaultValue),
    "ml_key": mlKey,
  ]
  if let scope = description.key.scope {
    object["scope"] = scope
  }
  if let numericConstraint = description.numericConstraint {
    object["numeric_constraint"] = cm_numeric_constraint_object(numericConstraint)
  }
  return object
}

func cm_model_description_object(_ description: MLModelDescription) -> [String: Any] {
  let inputs = description.inputDescriptionsByName.values
    .map(cm_feature_description_object)
    .sorted { ($0["name"] as? String ?? "") < ($1["name"] as? String ?? "") }
  let outputs = description.outputDescriptionsByName.values
    .map(cm_feature_description_object)
    .sorted { ($0["name"] as? String ?? "") < ($1["name"] as? String ?? "") }

  var stateFeatures: [[String: Any]] = []
  #if COREML_HAS_MACOS15_SDK
    if #available(macOS 15.0, *) {
      stateFeatures = description.stateDescriptionsByName.values
        .map(cm_feature_description_object)
        .sorted { ($0["name"] as? String ?? "") < ($1["name"] as? String ?? "") }
    }
  #endif

  let trainingInputs = description.trainingInputDescriptionsByName.values
    .map(cm_feature_description_object)
    .sorted { ($0["name"] as? String ?? "") < ($1["name"] as? String ?? "") }

  let parameterDescriptions = description.parameterDescriptionsByKey.values
    .map(cm_parameter_description_object)
    .sorted { ($0["key"] as? String ?? "") < ($1["key"] as? String ?? "") }

  var metadata: [String: Any] = [:]
  for (key, value) in description.metadata {
    metadata[key.rawValue] = cm_json_safe(value)
  }

  return [
    "inputs": inputs,
    "outputs": outputs,
    "state_features": stateFeatures,
    "training_inputs": trainingInputs,
    "parameter_descriptions": parameterDescriptions,
    "metadata": metadata,
    "predicted_feature_name": description.predictedFeatureName as Any,
    "predicted_probabilities_name": description.predictedProbabilitiesName as Any,
    "class_labels": (description.classLabels ?? []).map(cm_json_safe),
    "is_updatable": description.isUpdatable,
  ]
}

@_cdecl("cm_model_description_json")
public func cm_model_description_json(_ modelPtr: UnsafeMutableRawPointer?) -> UnsafeMutablePointer<
  CChar
>? {
  guard let modelPtr else {
    return cm_string("{}")
  }
  let model: MLModel = cm_borrow(modelPtr)
  return cm_string(cm_json_string(cm_model_description_object(model.modelDescription)))
}
