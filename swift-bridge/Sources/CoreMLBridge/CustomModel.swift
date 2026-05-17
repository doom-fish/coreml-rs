import CoreML
import Darwin
import Foundation
import ObjectiveC.runtime

@_silgen_name("cm_rust_custom_model_create")
private func cm_rust_custom_model_create(
  _ className: UnsafePointer<CChar>?,
  _ modelDescriptionJson: UnsafePointer<CChar>?,
  _ parametersJson: UnsafePointer<CChar>?,
  _ outContext: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32

@_silgen_name("cm_rust_custom_model_predict")
private func cm_rust_custom_model_predict(
  _ context: UnsafeMutableRawPointer?,
  _ input: UnsafeMutableRawPointer?,
  _ predictionOptionsJson: UnsafePointer<CChar>?,
  _ outProvider: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32

@_silgen_name("cm_rust_custom_model_predict_batch")
private func cm_rust_custom_model_predict_batch(
  _ context: UnsafeMutableRawPointer?,
  _ batch: UnsafeMutableRawPointer?,
  _ predictionOptionsJson: UnsafePointer<CChar>?,
  _ outBatch: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32

@_silgen_name("cm_rust_custom_model_release")
private func cm_rust_custom_model_release(_ context: UnsafeMutableRawPointer?)

private func cm_free_c_string(_ pointer: UnsafeMutablePointer<CChar>?) {
  guard let pointer else { return }
  free(pointer)
}

private func cm_take_rust_string(_ pointer: UnsafeMutablePointer<CChar>?) -> String {
  guard let pointer else {
    return "Rust custom-model bridge returned no error message"
  }
  defer { free(pointer) }
  return String(cString: pointer)
}

private func cm_throw_rust_error(_ errorPointer: UnsafeMutablePointer<CChar>?) throws -> Never {
  throw CMBridgeError.operationFailed(cm_take_rust_string(errorPointer))
}

private func cm_clone<T: AnyObject>(_ pointer: UnsafeMutableRawPointer) -> T {
  Unmanaged<T>.fromOpaque(pointer).retain().takeRetainedValue()
}

private func cm_is_subclass(_ candidate: AnyClass, of baseClass: AnyClass) -> Bool {
  var current: AnyClass? = candidate
  while let currentClass = current {
    if currentClass == baseClass {
      return true
    }
    current = class_getSuperclass(currentClass)
  }
  return false
}

private func cm_register_custom_model_class(named name: String) throws {
  if let existingClass = NSClassFromString(name) {
    guard cm_is_subclass(existingClass, of: CMRustCustomModelBase.self) else {
      throw CMBridgeError.operationFailed(
        "Objective-C class '\(name)' already exists and is not a Rust-backed MLCustomModel"
      )
    }
    return
  }

  let registered: AnyClass? = name.withCString { classNamePtr in
    guard let subclass = objc_allocateClassPair(CMRustCustomModelBase.self, classNamePtr, 0) else {
      return nil
    }
    objc_registerClassPair(subclass)
    return subclass
  }

  guard registered != nil else {
    throw CMBridgeError.operationFailed("failed to allocate Objective-C custom-model class '\(name)'")
  }
}

class CMRustCustomModelBase: NSObject, MLCustomModel {
  private var rustContext: UnsafeMutableRawPointer?

  required override init() {
    super.init()
  }

  fileprivate func configureRustContext(
    rustClassName: String,
    modelDescription: MLModelDescription,
    parameters: [String: Any]
  ) throws {
    let descriptionJson = cm_string(cm_json_string(cm_model_description_object(modelDescription)))
    let parametersJson = cm_string(cm_json_string(cm_json_safe(parameters)))
    defer {
      cm_free_c_string(descriptionJson)
      cm_free_c_string(parametersJson)
    }

    var errorPointer: UnsafeMutablePointer<CChar>?
    var context: UnsafeMutableRawPointer?
    let status = rustClassName.withCString { classNamePtr in
      cm_rust_custom_model_create(
        classNamePtr,
        descriptionJson,
        parametersJson,
        &context,
        &errorPointer
      )
    }
    guard status == CM_OK, let context else {
      try cm_throw_rust_error(errorPointer)
    }

    rustContext = context
  }

  required init(modelDescription: MLModelDescription, parameters: [String: Any]) throws {
    super.init()
    try configureRustContext(
      rustClassName: String(cString: object_getClassName(self)),
      modelDescription: modelDescription,
      parameters: parameters
    )
  }

  deinit {
    cm_rust_custom_model_release(rustContext)
  }

  private func retainedPredictionPointer(
    input: any MLFeatureProvider,
    options: MLPredictionOptions
  ) throws -> UnsafeMutableRawPointer {
    guard let rustContext else {
      throw CMBridgeError.operationFailed("custom-model instance was not initialized")
    }

    let inputBox = CMFeatureProviderBox(provider: input)
    let inputPointer = cm_retain(inputBox)
    defer { cm_object_release(inputPointer) }

    let optionsJson = cm_string(cm_json_string(cm_prediction_options_object(options)))
    defer { cm_free_c_string(optionsJson) }

    var errorPointer: UnsafeMutablePointer<CChar>?
    var outputPointer: UnsafeMutableRawPointer?
    let status = cm_rust_custom_model_predict(
      rustContext,
      inputPointer,
      optionsJson,
      &outputPointer,
      &errorPointer
    )
    guard status == CM_OK, let outputPointer else {
      try cm_throw_rust_error(errorPointer)
    }
    return outputPointer
  }

  private func retainedBatchPointer(
    inputBatch: any MLBatchProvider,
    options: MLPredictionOptions
  ) throws -> UnsafeMutableRawPointer {
    guard let rustContext else {
      throw CMBridgeError.operationFailed("custom-model instance was not initialized")
    }

    let batchBox = CMBatchProviderBox(batch: inputBatch)
    let batchPointer = cm_retain(batchBox)
    defer { cm_object_release(batchPointer) }

    let optionsJson = cm_string(cm_json_string(cm_prediction_options_object(options)))
    defer { cm_free_c_string(optionsJson) }

    var errorPointer: UnsafeMutablePointer<CChar>?
    var outputPointer: UnsafeMutableRawPointer?
    let status = cm_rust_custom_model_predict_batch(
      rustContext,
      batchPointer,
      optionsJson,
      &outputPointer,
      &errorPointer
    )
    guard status == CM_OK, let outputPointer else {
      try cm_throw_rust_error(errorPointer)
    }
    return outputPointer
  }

  func prediction(from input: any MLFeatureProvider, options: MLPredictionOptions) throws -> any MLFeatureProvider {
    let outputPointer = try retainedPredictionPointer(input: input, options: options)
    return Unmanaged<CMFeatureProviderBox>.fromOpaque(outputPointer).takeRetainedValue()
  }

  func predictions(from inputBatch: any MLBatchProvider, options: MLPredictionOptions) throws -> any MLBatchProvider {
    let outputPointer = try retainedBatchPointer(inputBatch: inputBatch, options: options)
    return Unmanaged<CMBatchProviderBox>.fromOpaque(outputPointer).takeRetainedValue()
  }
}

@_cdecl("cm_custom_model_register_class")
public func cm_custom_model_register_class(
  _ classNamePtr: UnsafePointer<CChar>?,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  autoreleasepool {
    guard let classNamePtr else {
      cm_write_error(errorOut, "custom-model class name must not be null")
      return CM_INVALID_ARGUMENT
    }

    do {
      try cm_register_custom_model_class(named: String(cString: classNamePtr))
      return CM_OK
    } catch {
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_CUSTOM_MODEL_FAILED)
    }
  }
}

@_cdecl("cm_custom_model_create")
public func cm_custom_model_create(
  _ classNamePtr: UnsafePointer<CChar>?,
  _ parametersJson: UnsafePointer<CChar>?,
  _ outModel: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  autoreleasepool {
    outModel.pointee = nil
    guard let classNamePtr else {
      cm_write_error(errorOut, "custom-model class name must not be null")
      return CM_INVALID_ARGUMENT
    }

    do {
      let className = String(cString: classNamePtr)
      try cm_register_custom_model_class(named: className)
      guard NSClassFromString(className) != nil else {
        throw CMBridgeError.operationFailed("registered custom-model class '\(className)' could not be resolved")
      }
      let parameters = try cm_json_object(from: parametersJson)
      let model = CMRustCustomModelBase()
      try model.configureRustContext(
        rustClassName: className,
        modelDescription: MLModelDescription(),
        parameters: parameters
      )
      outModel.pointee = cm_retain(model)
      return CM_OK
    } catch {
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_CUSTOM_MODEL_FAILED)
    }
  }
}

@_cdecl("cm_custom_model_predict")
public func cm_custom_model_predict(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ inputPtr: UnsafeMutableRawPointer?,
  _ predictionOptionsJson: UnsafePointer<CChar>?,
  _ outProvider: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  autoreleasepool {
    outProvider.pointee = nil
    guard let modelPtr, let inputPtr else {
      cm_write_error(errorOut, "custom-model prediction requires non-null model and feature-provider pointers")
      return CM_INVALID_ARGUMENT
    }

    let model: CMRustCustomModelBase = cm_clone(modelPtr)
    let input: CMFeatureProviderBox = cm_clone(inputPtr)

    do {
      let options = try cm_make_prediction_options(from: predictionOptionsJson)
      let output = try model.prediction(from: input, options: options)
      outProvider.pointee = cm_retain(CMFeatureProviderBox(provider: output))
      return CM_OK
    } catch {
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_CUSTOM_MODEL_FAILED)
    }
  }
}

@_cdecl("cm_custom_model_predict_batch")
public func cm_custom_model_predict_batch(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ batchPtr: UnsafeMutableRawPointer?,
  _ predictionOptionsJson: UnsafePointer<CChar>?,
  _ outBatch: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  autoreleasepool {
    outBatch.pointee = nil
    guard let modelPtr, let batchPtr else {
      cm_write_error(errorOut, "custom-model batch prediction requires non-null model and batch pointers")
      return CM_INVALID_ARGUMENT
    }

    let model: CMRustCustomModelBase = cm_clone(modelPtr)
    let batch: CMBatchProviderBox = cm_clone(batchPtr)

    do {
      let options = try cm_make_prediction_options(from: predictionOptionsJson)
      let output = try model.predictions(from: batch, options: options)
      outBatch.pointee = cm_retain(CMBatchProviderBox(batch: output))
      return CM_OK
    } catch {
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_CUSTOM_MODEL_FAILED)
    }
  }
}
