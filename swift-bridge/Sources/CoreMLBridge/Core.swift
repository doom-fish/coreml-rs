import CoreML
import CoreVideo
import Foundation

public let CM_OK: Int32 = 0
public let CM_INVALID_ARGUMENT: Int32 = -1
public let CM_MODEL_LOAD_FAILED: Int32 = -2
public let CM_PREDICTION_FAILED: Int32 = -3
public let CM_COMPILATION_FAILED: Int32 = -4
public let CM_FEATURE_PROVIDER_FAILED: Int32 = -5
public let CM_MULTI_ARRAY_FAILED: Int32 = -6
public let CM_UNSUPPORTED: Int32 = -7
public let CM_TIMED_OUT: Int32 = -8
public let CM_MODEL_ASSET_FAILED: Int32 = -9
public let CM_INDEX_OUT_OF_RANGE: Int32 = -10
public let CM_UNKNOWN: Int32 = -99

@inline(__always)
public func cm_retain<T: AnyObject>(_ object: T) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(object).toOpaque()
}

@inline(__always)
public func cm_release<T: AnyObject>(_ ptr: UnsafeMutableRawPointer, as _: T.Type) {
    Unmanaged<T>.fromOpaque(ptr).release()
}

@inline(__always)
public func cm_borrow<T: AnyObject>(_ ptr: UnsafeMutableRawPointer) -> T {
    Unmanaged<T>.fromOpaque(ptr).takeUnretainedValue()
}

@inline(__always)
public func cm_status(from error: Error) -> Int32 {
    Int32((error as NSError).code)
}

@_cdecl("cm_object_release")
public func cm_object_release(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr else { return }
    Unmanaged<AnyObject>.fromOpaque(ptr).release()
}

@inline(__always)
func cm_string(_ value: String) -> UnsafeMutablePointer<CChar>? {
    value.withCString { strdup($0) }
}

@inline(__always)
func cm_write_error(
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ message: String
) {
    errorOut?.pointee = cm_string(message)
}

@inline(__always)
func cm_url(from path: UnsafePointer<CChar>) -> URL {
    let raw = String(cString: path)
    if raw.hasPrefix("file:"), let url = URL(string: raw) {
        return url
    }
    return URL(fileURLWithPath: raw)
}

func cm_block_on_async<T>(
    timeoutSeconds: Int = 60,
    work: @escaping () async throws -> T,
    onSuccess: @escaping (T) -> Void
) -> Int32 {
    let semaphore = DispatchSemaphore(value: 0)
    var status: Int32 = CM_OK

    Task {
        defer { semaphore.signal() }
        do {
            let result = try await work()
            onSuccess(result)
        } catch {
            status = cm_status(from: error)
        }
    }

    if semaphore.wait(timeout: .now() + .seconds(timeoutSeconds)) == .timedOut {
        return CM_TIMED_OUT
    }

    return status
}

func cm_json_safe(_ value: Any) -> Any {
    switch value {
    case let dict as [String: Any]:
        return dict.mapValues(cm_json_safe)
    case let array as [Any]:
        return array.map(cm_json_safe)
    case let number as NSNumber:
        return number
    case let string as String:
        return string
    case let date as Date:
        return ISO8601DateFormatter().string(from: date)
    case _ as NSNull:
        return NSNull()
    default:
        return String(describing: value)
    }
}

func cm_json_string(_ value: Any) -> String {
    guard JSONSerialization.isValidJSONObject(value) else {
        return "{}"
    }

    do {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        return String(data: data, encoding: .utf8) ?? "{}"
    } catch {
        return "{}"
    }
}

func cm_multi_array_data_type(from rawValue: Int32) -> MLMultiArrayDataType? {
    switch rawValue {
    case 0x10000 | 64:
        return .float64
    case 0x10000 | 32:
        return .float32
    case 0x10000 | 16:
        return .float16
    case 0x20000 | 32:
        return .int32
    default:
        return nil
    }
}

func cm_multi_array_data_type_name(_ dataType: MLMultiArrayDataType) -> String {
    switch dataType {
    case .float64, .double:
        return "float64"
    case .float32, .float:
        return "float32"
    case .float16:
        return "float16"
    case .int32:
        return "int32"
    default:
        return "unknown"
    }
}

func cm_feature_type_name(_ featureType: MLFeatureType) -> String {
    switch featureType {
    case .invalid:
        return "invalid"
    case .int64:
        return "int64"
    case .double:
        return "double"
    case .string:
        return "string"
    case .image:
        return "image"
    case .multiArray:
        return "multi_array"
    case .dictionary:
        return "dictionary"
    case .sequence:
        return "sequence"
    case .state:
        return "state"
    @unknown default:
        return "invalid"
    }
}
