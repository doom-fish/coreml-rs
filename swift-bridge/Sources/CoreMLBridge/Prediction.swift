import CoreML
import Foundation

func cm_make_prediction_options(from jsonPtr: UnsafePointer<CChar>?) throws -> MLPredictionOptions {
    let object = try cm_json_object(from: jsonPtr)
    let options = MLPredictionOptions()
    if let usesCPUOnly = object["uses_cpu_only"] as? Bool {
        options.usesCPUOnly = usesCPUOnly
    }
    return options
}

func cm_prediction_options_object(_ options: MLPredictionOptions) -> [String: Any] {
    ["uses_cpu_only": options.usesCPUOnly]
}

@_cdecl("cm_prediction_options_snapshot_json")
public func cm_prediction_options_snapshot_json(
    _ predictionOptionsJson: UnsafePointer<CChar>?,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutablePointer<CChar>? {
    do {
        let options = try cm_make_prediction_options(from: predictionOptionsJson)
        return cm_string(cm_json_string(cm_prediction_options_object(options)))
    } catch {
        cm_write_error(errorOut, error.localizedDescription)
        return nil
    }
}
