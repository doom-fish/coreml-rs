import CoreML
import Foundation

#if COREML_HAS_MACOS14_4_SDK
  @available(macOS 14.4, *)
  func cm_model_structure_program_value_type_object(
    _: __MLModelStructureProgramValueType
  ) -> [String: Any] {
    ["description": "ValueType()"]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_program_value_object(
    _: __MLModelStructureProgramValue
  ) -> [String: Any] {
    ["description": "Value()"]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_program_named_value_type_object(
    _ namedValueType: __MLModelStructureProgramNamedValueType
  ) -> [String: Any] {
    [
      "name": namedValueType.name,
      "value_type": cm_model_structure_program_value_type_object(namedValueType.type),
    ]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_program_binding_object(
    _ binding: __MLModelStructureProgramBinding
  ) -> [String: Any] {
    if let value = binding.value {
      return ["value": cm_model_structure_program_value_object(value)]
    }
    return ["name": binding.name ?? ""]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_program_argument_object(
    _ argument: __MLModelStructureProgramArgument
  ) -> [String: Any] {
    [
      "bindings": argument.bindings.map(cm_model_structure_program_binding_object)
    ]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_program_operation_object(
    _ operation: __MLModelStructureProgramOperation
  ) -> [String: Any] {
    [
      "operator_name": operation.operatorName,
      "inputs": operation.inputs.mapValues { cm_model_structure_program_argument_object($0) },
      "outputs": operation.outputs.map(cm_model_structure_program_named_value_type_object),
      "blocks": operation.blocks.map(cm_model_structure_program_block_object),
    ]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_program_block_object(
    _ block: __MLModelStructureProgramBlock
  ) -> [String: Any] {
    [
      "inputs": block.inputs.map(cm_model_structure_program_named_value_type_object),
      "output_names": block.outputNames,
      "operations": block.operations.map(cm_model_structure_program_operation_object),
    ]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_program_function_object(
    _ function: __MLModelStructureProgramFunction
  ) -> [String: Any] {
    [
      "inputs": function.inputs.map(cm_model_structure_program_named_value_type_object),
      "block": cm_model_structure_program_block_object(function.block),
    ]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_program_object(_ program: __MLModelStructureProgram) -> [String: Any] {
    [
      "functions": program.functions.mapValues { cm_model_structure_program_function_object($0) }
    ]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_neural_network_layer_object(
    _ layer: __MLModelStructureNeuralNetworkLayer
  ) -> [String: Any] {
    [
      "name": layer.name,
      "type": layer.type,
      "input_names": layer.inputNames,
      "output_names": layer.outputNames,
    ]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_neural_network_object(
    _ neuralNetwork: __MLModelStructureNeuralNetwork
  ) -> [String: Any] {
    [
      "layers": neuralNetwork.layers.map(cm_model_structure_neural_network_layer_object)
    ]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_pipeline_object(_ pipeline: __MLModelStructurePipeline) -> [String: Any] {
    [
      "sub_model_names": pipeline.subModelNames,
      "sub_models": pipeline.subModels.map(cm_model_structure_object),
    ]
  }

  @available(macOS 14.4, *)
  func cm_model_structure_object(_ structure: __MLModelStructure) -> [String: Any] {
    if let neuralNetwork = structure.neuralNetwork {
      return [
        "kind": "neural_network",
        "neural_network": cm_model_structure_neural_network_object(neuralNetwork),
      ]
    }
    if let program = structure.program {
      return [
        "kind": "program",
        "program": cm_model_structure_program_object(program),
      ]
    }
    if let pipeline = structure.pipeline {
      return [
        "kind": "pipeline",
        "pipeline": cm_model_structure_pipeline_object(pipeline),
      ]
    }
    return ["kind": "unsupported"]
  }

  @_cdecl("cm_model_structure_load_json")
  public func cm_model_structure_load_json(
    _ pathPtr: UnsafePointer<CChar>?,
    _ timeoutSeconds: Double,
    _ outJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
  ) -> Int32 {
    outJson.pointee = nil
    guard let pathPtr else {
      cm_write_error(errorOut, "model-structure path must not be null")
      return CM_INVALID_ARGUMENT
    }
    if #available(macOS 14.4, *) {
      let url = cm_url(from: pathPtr)
      switch cm_block_on_async(timeoutSeconds: timeoutSeconds, work: {
        try await cm_await_completion { __MLModelStructure.loadContents(of: url, completionHandler: $0) }
      }) {
      case .success(let structure):
        outJson.pointee = cm_string(cm_json_string(cm_model_structure_object(structure)))
        return CM_OK
      case .failure(let error):
        cm_write_error(errorOut, error.localizedDescription)
        return cm_status_code(for: error, fallback: CM_DESCRIPTION_FAILED)
      }
    }
    cm_write_error(errorOut, "MLModelStructure requires macOS 14.4+")
    return CM_UNSUPPORTED
  }
#else
  @_cdecl("cm_model_structure_load_json")
  public func cm_model_structure_load_json(
    _: UnsafePointer<CChar>?,
    _: Double,
    _ outJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
  ) -> Int32 {
    outJson.pointee = nil
    cm_write_error(errorOut, "MLModelStructure requires a macOS 14.4+ SDK")
    return CM_UNSUPPORTED
  }
#endif
