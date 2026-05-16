import CoreML
import Foundation

func cm_compute_units(from name: String?) -> MLComputeUnits {
    switch name {
    case "cpu_only":
        return .cpuOnly
    case "cpu_and_gpu":
        return .cpuAndGPU
    case "cpu_and_neural_engine":
        return .cpuAndNeuralEngine
    default:
        return .all
    }
}

func cm_compute_units_name(_ computeUnits: MLComputeUnits) -> String {
    switch computeUnits {
    case .cpuOnly:
        return "cpu_only"
    case .cpuAndGPU:
        return "cpu_and_gpu"
    case .cpuAndNeuralEngine:
        return "cpu_and_neural_engine"
    default:
        return "all"
    }
}

func cm_make_configuration(from jsonPtr: UnsafePointer<CChar>?) throws -> MLModelConfiguration {
    let object = try cm_json_object(from: jsonPtr)
    let configuration = MLModelConfiguration()
    configuration.computeUnits = cm_compute_units(from: object["compute_units"] as? String)

    if let allowLowPrecision = object["allow_low_precision_accumulation_on_gpu"] as? Bool {
        configuration.allowLowPrecisionAccumulationOnGPU = allowLowPrecision
    }
    if let displayName = object["display_name"] as? String {
        configuration.modelDisplayName = displayName
    }

    #if COREML_HAS_MACOS15_SDK
        if #available(macOS 15.0, *), let functionName = object["function_name"] as? String {
            configuration.functionName = functionName
        }
    #endif

    #if COREML_HAS_MACOS14_4_SDK
        if #available(macOS 14.4, *), let hintsObject = object["optimization_hints"] as? [String: Any] {
            var hints = MLOptimizationHints()
            if hintsObject["reshape_frequency"] as? String == "infrequent" {
                hints.reshapeFrequency = .infrequent
            } else {
                hints.reshapeFrequency = .frequent
            }
            #if COREML_HAS_MACOS15_SDK
                if #available(macOS 15.0, *), hintsObject["specialization_strategy"] as? String == "fast_prediction" {
                    hints.specializationStrategy = .fastPrediction
                }
            #endif
            configuration.optimizationHints = hints
        }
    #endif

    if let parameterObject = object["parameters"] as? [String: Any] {
        var parameters: [MLParameterKey: Any] = [:]
        for (keyName, value) in parameterObject {
            guard let parameterKey = cm_parameter_key(named: keyName) else {
                throw CMBridgeError.invalidArgument("unsupported MLParameterKey '\(keyName)'")
            }
            if let stringValue = value as? String {
                parameters[parameterKey] = stringValue
            } else if let numberValue = value as? NSNumber {
                parameters[parameterKey] = numberValue
            } else {
                throw CMBridgeError.invalidArgument(
                    "parameter '\(keyName)' must be encoded as a string, bool, integer, or double"
                )
            }
        }
        configuration.parameters = parameters
    }

    return configuration
}

func cm_model_configuration_object(_ configuration: MLModelConfiguration) -> [String: Any] {
    var object: [String: Any] = [
        "compute_units": cm_compute_units_name(configuration.computeUnits),
        "allow_low_precision_accumulation_on_gpu": configuration.allowLowPrecisionAccumulationOnGPU
    ]

    if let displayName = configuration.modelDisplayName {
        object["display_name"] = displayName
    }

    #if COREML_HAS_MACOS15_SDK
        if #available(macOS 15.0, *), let functionName = configuration.functionName {
            object["function_name"] = functionName
        }
    #endif

    #if COREML_HAS_MACOS14_4_SDK
        if #available(macOS 14.4, *) {
            let optimizationHints = configuration.optimizationHints
            var hintsObject: [String: Any] = [
                "reshape_frequency": optimizationHints.reshapeFrequency == .infrequent ? "infrequent" : "frequent"
            ]
            #if COREML_HAS_MACOS15_SDK
                if #available(macOS 15.0, *) {
                    hintsObject["specialization_strategy"] =
                        optimizationHints.specializationStrategy == .fastPrediction ? "fast_prediction" : "default"
                }
            #endif
            object["optimization_hints"] = hintsObject
        }
    #endif

    if let parameters = configuration.parameters {
        var parameterObject: [String: Any] = [:]
        for (key, value) in parameters {
            let parameterName = key.scope.map { "\(key.name):\($0)" } ?? key.name
            parameterObject[parameterName] = cm_json_safe(value)
        }
        object["parameters"] = parameterObject
    }

    return object
}

@_cdecl("cm_model_configuration_snapshot_json")
public func cm_model_configuration_snapshot_json(
    _ configurationJson: UnsafePointer<CChar>?,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutablePointer<CChar>? {
    do {
        let configuration = try cm_make_configuration(from: configurationJson)
        return cm_string(cm_json_string(cm_model_configuration_object(configuration)))
    } catch {
        cm_write_error(errorOut, error.localizedDescription)
        return nil
    }
}
