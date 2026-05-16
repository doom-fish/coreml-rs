import CoreML
import Foundation

#if COREML_HAS_MACOS14_4_SDK
    @available(macOS 14.4, *)
    func cm_compute_plan_object(_ plan: MLComputePlan) -> [String: Any] {
        _ = plan
        return [
            "model_type": "unknown",
            "function_names": [],
            "operation_count": 0,
            "layer_count": 0,
            "pipeline_model_count": 0
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
                    try await MLComputePlan.load(contentsOf: cm_url(from: pathPtr), configuration: configuration)
                }) {
                case let .success(plan):
                    outSummaryJson.pointee = cm_string(cm_json_string(cm_compute_plan_object(plan)))
                    return CM_OK
                case let .failure(error):
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
