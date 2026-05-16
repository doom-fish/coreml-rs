import CoreML
import Foundation

func cm_compute_units(from rawValue: Int32) -> MLComputeUnits {
    switch rawValue {
    case 0:
        return .cpuOnly
    case 1:
        return .cpuAndGPU
    case 3:
        return .cpuAndNeuralEngine
    default:
        return .all
    }
}

func cm_make_configuration(
    computeUnits: Int32,
    allowLowPrecisionAccumulationOnGPU: Bool,
    modelDisplayName: UnsafePointer<CChar>?
) -> MLModelConfiguration {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = cm_compute_units(from: computeUnits)
    configuration.allowLowPrecisionAccumulationOnGPU = allowLowPrecisionAccumulationOnGPU
    if let modelDisplayName {
        configuration.modelDisplayName = String(cString: modelDisplayName)
    }
    return configuration
}
