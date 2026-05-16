import CoreML
import Foundation

#if COREML_HAS_MACOS14_4_SDK
  @available(macOS 14.4, *)
  func cm_compute_plan_cost_object(_ cost: MLComputePlan.Cost) -> [String: Any] {
    ["weight": cost.weight]
  }

  @available(macOS 14.4, *)
  func cm_compute_plan_device_usage_object(
    _ usage: MLComputePlan.DeviceUsage
  ) -> [String: Any] {
    [
      "supported": usage.supported.map { cm_compute_device_object($0) },
      "preferred": cm_compute_device_object(usage.preferred),
    ]
  }

  @available(macOS 14.4, *)
  func cm_collect_compute_plan_program_operation_plans(
    _ block: MLModelStructure.Program.Block,
    path: String,
    plan: MLComputePlan
  ) -> [[String: Any]] {
    var objects: [[String: Any]] = []
    for (index, operation) in block.operations.enumerated() {
      let operationPath = "\(path).operations[\(index)]"
      var object: [String: Any] = [
        "path": operationPath,
        "operation": cm_model_structure_program_operation_object(operation),
      ]
      if let cost = plan.estimatedCost(of: operation) {
        object["cost"] = cm_compute_plan_cost_object(cost)
      }
      if let usage = plan.deviceUsage(for: operation) {
        object["device_usage"] = cm_compute_plan_device_usage_object(usage)
      }
      objects.append(object)

      for (blockIndex, nestedBlock) in operation.blocks.enumerated() {
        objects.append(
          contentsOf: cm_collect_compute_plan_program_operation_plans(
            nestedBlock,
            path: "\(operationPath).blocks[\(blockIndex)]",
            plan: plan
          )
        )
      }
    }
    return objects
  }

  @available(macOS 14.4, *)
  func cm_collect_compute_plan_program_operation_plans(
    _ program: MLModelStructure.Program,
    plan: MLComputePlan
  ) -> [[String: Any]] {
    var objects: [[String: Any]] = []
    for functionName in program.functions.keys.sorted() {
      guard let function = program.functions[functionName] else { continue }
      objects.append(
        contentsOf: cm_collect_compute_plan_program_operation_plans(
          function.block,
          path: "functions.\(functionName).block",
          plan: plan
        )
      )
    }
    return objects
  }

  @available(macOS 14.4, *)
  func cm_collect_compute_plan_neural_network_layer_plans(
    _ neuralNetwork: MLModelStructure.NeuralNetwork,
    plan: MLComputePlan
  ) -> [[String: Any]] {
    neuralNetwork.layers.enumerated().map { index, layer in
      var object: [String: Any] = [
        "path": "layers[\(index)]",
        "layer": cm_model_structure_neural_network_layer_object(layer),
      ]
      if let usage = plan.deviceUsage(for: layer) {
        object["device_usage"] = cm_compute_plan_device_usage_object(usage)
      }
      return object
    }
  }

  @available(macOS 14.4, *)
  func cm_compute_plan_object(_ plan: MLComputePlan) -> [String: Any] {
    let modelStructureObject = cm_model_structure_object(plan.modelStructure)

    let modelType: String
    let functionNames: [String]
    let programOperationPlans: [[String: Any]]
    let neuralNetworkLayerPlans: [[String: Any]]
    let layerCount: Int
    let pipelineModelCount: Int

    switch plan.modelStructure {
    case .program(let program):
      modelType = "program"
      functionNames = program.functions.keys.sorted()
      programOperationPlans = cm_collect_compute_plan_program_operation_plans(program, plan: plan)
      neuralNetworkLayerPlans = []
      layerCount = 0
      pipelineModelCount = 0
    case .neuralNetwork(let neuralNetwork):
      modelType = "neural_network"
      functionNames = []
      programOperationPlans = []
      neuralNetworkLayerPlans = cm_collect_compute_plan_neural_network_layer_plans(
        neuralNetwork,
        plan: plan
      )
      layerCount = neuralNetwork.layers.count
      pipelineModelCount = 0
    case .pipeline(let pipeline):
      modelType = "pipeline"
      functionNames = []
      programOperationPlans = []
      neuralNetworkLayerPlans = []
      layerCount = 0
      pipelineModelCount = pipeline.subModels.count
    case .unsupported:
      modelType = "unknown"
      functionNames = []
      programOperationPlans = []
      neuralNetworkLayerPlans = []
      layerCount = 0
      pipelineModelCount = 0
    @unknown default:
      modelType = "unknown"
      functionNames = []
      programOperationPlans = []
      neuralNetworkLayerPlans = []
      layerCount = 0
      pipelineModelCount = 0
    }

    return [
      "model_type": modelType,
      "function_names": functionNames,
      "operation_count": programOperationPlans.count,
      "layer_count": layerCount,
      "pipeline_model_count": pipelineModelCount,
      "model_structure": modelStructureObject,
      "program_operation_plans": programOperationPlans,
      "neural_network_layer_plans": neuralNetworkLayerPlans,
    ]
  }

  @_cdecl("cm_compute_plan_load_summary")
  public func cm_compute_plan_load_summary(
    _ pathPtr: UnsafePointer<CChar>?,
    _ configurationJson: UnsafePointer<CChar>?,
    _ outSummaryJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
  ) -> Int32 {
    outSummaryJson.pointee = nil
    guard let pathPtr else {
      cm_write_error(errorOut, "compute-plan path must not be null")
      return CM_INVALID_ARGUMENT
    }
    if #available(macOS 14.4, *) {
      do {
        let configuration = try cm_make_configuration(from: configurationJson)
        switch cm_block_on_async(work: {
          try await MLComputePlan.load(
            contentsOf: cm_url(from: pathPtr), configuration: configuration)
        }) {
        case .success(let plan):
          outSummaryJson.pointee = cm_string(cm_json_string(cm_compute_plan_object(plan)))
          return CM_OK
        case .failure(let error):
          cm_write_error(errorOut, error.localizedDescription)
          return cm_status_code(for: error, fallback: CM_COMPUTE_PLAN_FAILED)
        }
      } catch {
        cm_write_error(errorOut, error.localizedDescription)
        return cm_status_code(for: error, fallback: CM_COMPUTE_PLAN_FAILED)
      }
    }
    cm_write_error(errorOut, "MLComputePlan requires macOS 14.4+")
    return CM_UNSUPPORTED
  }
#else
  @_cdecl("cm_compute_plan_load_summary")
  public func cm_compute_plan_load_summary(
    _: UnsafePointer<CChar>?,
    _: UnsafePointer<CChar>?,
    _ outSummaryJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
  ) -> Int32 {
    outSummaryJson.pointee = nil
    cm_write_error(errorOut, "MLComputePlan requires a macOS 14.4+ SDK")
    return CM_UNSUPPORTED
  }
#endif
