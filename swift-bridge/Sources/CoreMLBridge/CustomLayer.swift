import CoreML
import Darwin
import Foundation
import Metal
import ObjectiveC.runtime

private let CM_CUSTOM_LAYER_GPU_ENCODING_FLAG: UInt32 = 1

@_silgen_name("cm_rust_custom_layer_create")
private func cm_rust_custom_layer_create(
  _ className: UnsafePointer<CChar>?,
  _ parametersJson: UnsafePointer<CChar>?,
  _ outContext: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ outFlags: UnsafeMutablePointer<UInt32>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32

@_silgen_name("cm_rust_custom_layer_set_weight_data")
private func cm_rust_custom_layer_set_weight_data(
  _ context: UnsafeMutableRawPointer?,
  _ weightData: UnsafePointer<UnsafePointer<UInt8>?>?,
  _ weightLengths: UnsafePointer<Int>?,
  _ weightCount: Int,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32

@_silgen_name("cm_rust_custom_layer_output_shapes_json")
private func cm_rust_custom_layer_output_shapes_json(
  _ context: UnsafeMutableRawPointer?,
  _ inputShapesJson: UnsafePointer<CChar>?,
  _ outJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32

@_silgen_name("cm_rust_custom_layer_evaluate_cpu")
private func cm_rust_custom_layer_evaluate_cpu(
  _ context: UnsafeMutableRawPointer?,
  _ inputs: UnsafePointer<UnsafeMutableRawPointer?>?,
  _ inputCount: Int,
  _ outputs: UnsafePointer<UnsafeMutableRawPointer?>?,
  _ outputCount: Int,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32

@_silgen_name("cm_rust_custom_layer_encode")
private func cm_rust_custom_layer_encode(
  _ context: UnsafeMutableRawPointer?,
  _ commandBuffer: UnsafeMutableRawPointer?,
  _ inputs: UnsafePointer<UnsafeMutableRawPointer?>?,
  _ inputCount: Int,
  _ outputs: UnsafePointer<UnsafeMutableRawPointer?>?,
  _ outputCount: Int,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32

@_silgen_name("cm_rust_custom_layer_release")
private func cm_rust_custom_layer_release(_ context: UnsafeMutableRawPointer?)

private func cm_free_c_string(_ pointer: UnsafeMutablePointer<CChar>?) {
  guard let pointer else { return }
  free(pointer)
}

private func cm_take_rust_string(_ pointer: UnsafeMutablePointer<CChar>?) -> String {
  guard let pointer else {
    return "Rust custom-layer bridge returned no error message"
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

private func cm_register_custom_layer_class(named name: String) throws {
  if let existingClass = NSClassFromString(name) {
    guard cm_is_subclass(existingClass, of: CMRustCustomLayerBase.self) else {
      throw CMBridgeError.operationFailed(
        "Objective-C class '\(name)' already exists and is not a Rust-backed MLCustomLayer"
      )
    }
    return
  }

  let registered: AnyClass? = name.withCString { classNamePtr in
    guard let subclass = objc_allocateClassPair(CMRustCustomLayerBase.self, classNamePtr, 0) else {
      return nil
    }
    objc_registerClassPair(subclass)
    return subclass
  }

  guard registered != nil else {
    throw CMBridgeError.operationFailed("failed to allocate Objective-C custom-layer class '\(name)'")
  }
}

class CMRustCustomLayerBase: NSObject, MLCustomLayer {
  private var rustContext: UnsafeMutableRawPointer?
  private var rustFlags: UInt32 = 0

  required override init() {
    super.init()
  }

  fileprivate func configureRustContext(rustClassName: String, parameters: [String: Any]) throws {
    let parametersJson = cm_string(cm_json_string(cm_json_safe(parameters)))
    defer { cm_free_c_string(parametersJson) }

    var errorPointer: UnsafeMutablePointer<CChar>?
    var context: UnsafeMutableRawPointer?
    var flags: UInt32 = 0
    let status = rustClassName.withCString { classNamePtr in
      cm_rust_custom_layer_create(classNamePtr, parametersJson, &context, &flags, &errorPointer)
    }
    guard status == CM_OK, let context else {
      try cm_throw_rust_error(errorPointer)
    }

    rustContext = context
    rustFlags = flags
  }

  required init(parameters: [String: Any]) throws {
    super.init()
    try configureRustContext(
      rustClassName: String(cString: object_getClassName(self)),
      parameters: parameters
    )
  }

  deinit {
    cm_rust_custom_layer_release(rustContext)
  }

  override func responds(to selector: Selector!) -> Bool {
    if selector == NSSelectorFromString("encodeToCommandBuffer:inputs:outputs:error:") {
      return (rustFlags & CM_CUSTOM_LAYER_GPU_ENCODING_FLAG) != 0
    }
    return super.responds(to: selector)
  }

  func setWeightData(_ weights: [Data]) throws {
    guard let rustContext else {
      throw CMBridgeError.operationFailed("custom-layer instance was not initialized")
    }

    let nsWeights = weights.map { $0 as NSData }
    let weightPointers: [UnsafePointer<UInt8>?] = nsWeights.map { nsData in
      nsData.bytes.assumingMemoryBound(to: UInt8.self)
    }
    let weightLengths = nsWeights.map(\.length)

    var errorPointer: UnsafeMutablePointer<CChar>?
    let status = cm_rust_custom_layer_set_weight_data(
      rustContext,
      weightPointers,
      weightLengths,
      nsWeights.count,
      &errorPointer
    )
    guard status == CM_OK else {
      try cm_throw_rust_error(errorPointer)
    }
  }

  func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
    guard let rustContext else {
      throw CMBridgeError.operationFailed("custom-layer instance was not initialized")
    }

    let inputShapesJson = cm_string(cm_json_string(inputShapes.map { $0.map(\.intValue) }))
    defer { cm_free_c_string(inputShapesJson) }

    var errorPointer: UnsafeMutablePointer<CChar>?
    var outputJson: UnsafeMutablePointer<CChar>?
    let status = cm_rust_custom_layer_output_shapes_json(
      rustContext,
      inputShapesJson,
      &outputJson,
      &errorPointer
    )
    guard status == CM_OK, let outputJson else {
      try cm_throw_rust_error(errorPointer)
    }

    let outputShapeString = cm_take_rust_string(outputJson)
    guard let data = outputShapeString.data(using: .utf8),
      let outputShapes = try JSONSerialization.jsonObject(with: data) as? [[Int]]
    else {
      throw CMBridgeError.operationFailed("failed to decode Rust custom-layer output-shape JSON")
    }
    return outputShapes.map { $0.map(NSNumber.init(value:)) }
  }

  func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
    guard let rustContext else {
      throw CMBridgeError.operationFailed("custom-layer instance was not initialized")
    }

    let inputPointers: [UnsafeMutableRawPointer?] = inputs.map(cm_retain)
    let outputPointers: [UnsafeMutableRawPointer?] = outputs.map(cm_retain)
    defer {
      inputPointers.forEach(cm_object_release)
      outputPointers.forEach(cm_object_release)
    }

    var errorPointer: UnsafeMutablePointer<CChar>?
    let status = cm_rust_custom_layer_evaluate_cpu(
      rustContext,
      inputPointers,
      inputPointers.count,
      outputPointers,
      outputPointers.count,
      &errorPointer
    )
    guard status == CM_OK else {
      try cm_throw_rust_error(errorPointer)
    }
  }

  func encode(
    commandBuffer: any MTLCommandBuffer,
    inputs: [any MTLTexture],
    outputs: [any MTLTexture]
  ) throws {
    guard let rustContext else {
      throw CMBridgeError.operationFailed("custom-layer instance was not initialized")
    }

    let commandBufferPointer = Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque()
    let inputPointers: [UnsafeMutableRawPointer?] = inputs.map {
      Unmanaged.passUnretained($0 as AnyObject).toOpaque()
    }
    let outputPointers: [UnsafeMutableRawPointer?] = outputs.map {
      Unmanaged.passUnretained($0 as AnyObject).toOpaque()
    }

    var errorPointer: UnsafeMutablePointer<CChar>?
    let status = cm_rust_custom_layer_encode(
      rustContext,
      commandBufferPointer,
      inputPointers,
      inputPointers.count,
      outputPointers,
      outputPointers.count,
      &errorPointer
    )
    guard status == CM_OK else {
      try cm_throw_rust_error(errorPointer)
    }
  }
}

@_cdecl("cm_custom_layer_register_class")
public func cm_custom_layer_register_class(
  _ classNamePtr: UnsafePointer<CChar>?,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  autoreleasepool {
    guard let classNamePtr else {
      cm_write_error(errorOut, "custom-layer class name must not be null")
      return CM_INVALID_ARGUMENT
    }

    do {
      try cm_register_custom_layer_class(named: String(cString: classNamePtr))
      return CM_OK
    } catch {
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_CUSTOM_LAYER_FAILED)
    }
  }
}

@_cdecl("cm_custom_layer_create")
public func cm_custom_layer_create(
  _ classNamePtr: UnsafePointer<CChar>?,
  _ parametersJson: UnsafePointer<CChar>?,
  _ outLayer: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  autoreleasepool {
    outLayer.pointee = nil
    guard let classNamePtr else {
      cm_write_error(errorOut, "custom-layer class name must not be null")
      return CM_INVALID_ARGUMENT
    }

    do {
      let className = String(cString: classNamePtr)
      try cm_register_custom_layer_class(named: className)
      guard NSClassFromString(className) != nil else {
        throw CMBridgeError.operationFailed("registered custom-layer class '\(className)' could not be resolved")
      }
      let parameters = try cm_json_object(from: parametersJson)
      let layer = CMRustCustomLayerBase()
      try layer.configureRustContext(rustClassName: className, parameters: parameters)
      outLayer.pointee = cm_retain(layer)
      return CM_OK
    } catch {
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_CUSTOM_LAYER_FAILED)
    }
  }
}

@_cdecl("cm_custom_layer_set_weight_data")
public func cm_custom_layer_set_weight_data(
  _ layerPtr: UnsafeMutableRawPointer?,
  _ weightData: UnsafePointer<UnsafePointer<UInt8>?>?,
  _ weightLengths: UnsafePointer<Int>?,
  _ weightCount: Int,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  autoreleasepool {
    guard let layerPtr else {
      cm_write_error(errorOut, "custom-layer instance must not be null")
      return CM_INVALID_ARGUMENT
    }
    if weightCount > 0 && (weightData == nil || weightLengths == nil) {
      cm_write_error(errorOut, "custom-layer weights require non-null data and length arrays")
      return CM_INVALID_ARGUMENT
    }

    let layer: CMRustCustomLayerBase = cm_clone(layerPtr)
    let weightPointers = weightCount > 0 ? Array(UnsafeBufferPointer(start: weightData, count: weightCount)) : []
    let lengths = weightCount > 0 ? Array(UnsafeBufferPointer(start: weightLengths, count: weightCount)) : []

    do {
      let weights = try zip(weightPointers, lengths).enumerated().map { index, pair in
        let (bytes, length) = pair
        if length == 0 {
          return Data()
        }
        guard let bytes else {
          throw CMBridgeError.invalidArgument(
            "custom-layer weight blob \(index) was null despite having a non-zero length"
          )
        }
        return Data(bytes: bytes, count: length)
      }
      try layer.setWeightData(weights)
      return CM_OK
    } catch {
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_CUSTOM_LAYER_FAILED)
    }
  }
}

@_cdecl("cm_custom_layer_output_shapes_json")
public func cm_custom_layer_output_shapes_json(
  _ layerPtr: UnsafeMutableRawPointer?,
  _ inputShapesJson: UnsafePointer<CChar>?,
  _ outJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  autoreleasepool {
    outJson.pointee = nil
    guard let layerPtr else {
      cm_write_error(errorOut, "custom-layer instance must not be null")
      return CM_INVALID_ARGUMENT
    }

    let layer: CMRustCustomLayerBase = cm_clone(layerPtr)

    do {
      let json = inputShapesJson.map { String(cString: $0) } ?? "[]"
      let data = json.data(using: String.Encoding.utf8) ?? Data()
      let rawInputShapes = try JSONSerialization.jsonObject(with: data) as? [[Int]] ?? []
      let inputShapes = rawInputShapes.map { shape in shape.map(NSNumber.init(value:)) }
      let outputShapes = try layer.outputShapes(forInputShapes: inputShapes)
      outJson.pointee = cm_string(cm_json_string(outputShapes.map { shape in shape.map { $0.intValue } }))
      return CM_OK
    } catch {
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_CUSTOM_LAYER_FAILED)
    }
  }
}

@_cdecl("cm_custom_layer_evaluate_cpu")
public func cm_custom_layer_evaluate_cpu(
  _ layerPtr: UnsafeMutableRawPointer?,
  _ inputPointers: UnsafePointer<UnsafeMutableRawPointer?>?,
  _ inputCount: Int,
  _ outputPointers: UnsafePointer<UnsafeMutableRawPointer?>?,
  _ outputCount: Int,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  autoreleasepool {
    guard let layerPtr else {
      cm_write_error(errorOut, "custom-layer instance must not be null")
      return CM_INVALID_ARGUMENT
    }
    if inputCount > 0 && inputPointers == nil {
      cm_write_error(errorOut, "custom-layer input pointers must not be null when count is non-zero")
      return CM_INVALID_ARGUMENT
    }
    if outputCount > 0 && outputPointers == nil {
      cm_write_error(errorOut, "custom-layer output pointers must not be null when count is non-zero")
      return CM_INVALID_ARGUMENT
    }

    let layer: CMRustCustomLayerBase = cm_clone(layerPtr)
    let rawInputs = inputCount > 0 ? Array(UnsafeBufferPointer(start: inputPointers, count: inputCount)) : []
    let rawOutputs = outputCount > 0 ? Array(UnsafeBufferPointer(start: outputPointers, count: outputCount)) : []

    do {
      let inputs = try rawInputs.enumerated().map { index, pointer -> MLMultiArray in
        guard let pointer else {
          throw CMBridgeError.invalidArgument("custom-layer input \(index) was null")
        }
        let array: MLMultiArray = cm_clone(pointer)
        return array
      }
      let outputs = try rawOutputs.enumerated().map { index, pointer -> MLMultiArray in
        guard let pointer else {
          throw CMBridgeError.invalidArgument("custom-layer output \(index) was null")
        }
        let array: MLMultiArray = cm_clone(pointer)
        return array
      }
      try layer.evaluate(inputs: inputs, outputs: outputs)
      return CM_OK
    } catch {
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_CUSTOM_LAYER_FAILED)
    }
  }
}
