import CoreML
import Foundation

#if COREML_HAS_MACOS14_SDK || COREML_HAS_MACOS14_4_SDK
  @available(macOS 14.0, *)
  func cm_compute_device_object(_ device: MLComputeDevice) -> [String: Any] {
    switch device {
    case .cpu:
      return [
        "kind": "cpu",
        "description": device.description,
      ]
    case .gpu:
      return [
        "kind": "gpu",
        "description": device.description,
      ]
    case .neuralEngine(let neuralEngine):
      return [
        "kind": "neural_engine",
        "description": device.description,
        "total_core_count": neuralEngine.totalCoreCount,
      ]
    @unknown default:
      return [
        "kind": "unknown",
        "description": device.description,
      ]
    }
  }

  @available(macOS 14.0, *)
  func cm_compute_devices_json(_ devices: [MLComputeDevice]) -> UnsafeMutablePointer<CChar>? {
    cm_string(cm_json_string(devices.map(cm_compute_device_object)))
  }

  @_cdecl("cm_all_compute_devices_json")
  public func cm_all_compute_devices_json(
    _ outJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
  ) -> Int32 {
    outJson.pointee = nil
    if #available(macOS 14.0, *) {
      outJson.pointee = cm_compute_devices_json(MLComputeDevice.allComputeDevices)
      return CM_OK
    }
    cm_write_error(errorOut, "CoreML compute-device discovery requires macOS 14.0+")
    return CM_UNSUPPORTED
  }

  @_cdecl("cm_model_available_compute_devices_json")
  public func cm_model_available_compute_devices_json(
    _ outJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
  ) -> Int32 {
    outJson.pointee = nil
    if #available(macOS 14.0, *) {
      outJson.pointee = cm_compute_devices_json(MLModel.availableComputeDevices)
      return CM_OK
    }
    cm_write_error(errorOut, "MLModel.availableComputeDevices requires macOS 14.0+")
    return CM_UNSUPPORTED
  }
#else
  @_cdecl("cm_all_compute_devices_json")
  public func cm_all_compute_devices_json(
    _ outJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
  ) -> Int32 {
    outJson.pointee = nil
    cm_write_error(errorOut, "CoreML compute-device discovery requires a macOS 14.0+ SDK")
    return CM_UNSUPPORTED
  }

  @_cdecl("cm_model_available_compute_devices_json")
  public func cm_model_available_compute_devices_json(
    _ outJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
  ) -> Int32 {
    outJson.pointee = nil
    cm_write_error(errorOut, "MLModel.availableComputeDevices requires a macOS 14.0+ SDK")
    return CM_UNSUPPORTED
  }
#endif
