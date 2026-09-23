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
public let CM_DESCRIPTION_FAILED: Int32 = -11
public let CM_COMPUTE_PLAN_FAILED: Int32 = -12
public let CM_UPDATE_FAILED: Int32 = -13
public let CM_STATE_FAILED: Int32 = -14
public let CM_CUSTOM_LAYER_FAILED: Int32 = -15
public let CM_CUSTOM_MODEL_FAILED: Int32 = -16
public let CM_UNKNOWN: Int32 = -99

enum CMBridgeError: LocalizedError {
    case timedOut(String)
    case invalidArgument(String)
    case unsupported(String)
    case operationFailed(String)

    var errorDescription: String? {
        switch self {
        case let .timedOut(message):
            return message
        case let .invalidArgument(message):
            return message
        case let .unsupported(message):
            return message
        case let .operationFailed(message):
            return message
        }
    }
}

final class CMAsyncResult<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Result<T, Error>?
    private var abandoned = false
    private let discard: (T) -> Void

    init(discard: @escaping (T) -> Void) {
        self.discard = discard
    }

    func store(_ result: Result<T, Error>) {
        lock.lock()
        guard !abandoned else {
            lock.unlock()
            if case let .success(value) = result {
                discard(value)
            }
            return
        }
        stored = result
        lock.unlock()
    }

    func take() -> Result<T, Error>? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }

    func abandon() {
        lock.lock()
        abandoned = true
        let late = stored
        stored = nil
        lock.unlock()
        if case let .success(value) = late {
            discard(value)
        }
    }
}

final class CMTaskHandle: NSObject, @unchecked Sendable {
    private let cancelWork: () -> Void

    init<Success, Failure>(_ task: Task<Success, Failure>) {
        cancelWork = { task.cancel() }
        super.init()
    }

    func cancel() {
        cancelWork()
    }
}

@_cdecl("cm_task_cancel")
public func cm_task_cancel(_ handlePtr: UnsafeMutableRawPointer?) {
    guard let handlePtr else { return }
    let handle: CMTaskHandle = cm_borrow(handlePtr)
    handle.cancel()
}

@inline(__always)
public func cm_retain<T: AnyObject>(_ object: T) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(object).toOpaque()
}

@inline(__always)
public func cm_borrow<T: AnyObject>(_ ptr: UnsafeMutableRawPointer) -> T {
    Unmanaged<T>.fromOpaque(ptr).takeUnretainedValue()
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
func cm_status_code(for error: Error, fallback: Int32) -> Int32 {
    guard let bridgeError = error as? CMBridgeError else {
        return fallback
    }
    switch bridgeError {
    case .timedOut:
        return CM_TIMED_OUT
    case .invalidArgument:
        return CM_INVALID_ARGUMENT
    case .unsupported:
        return CM_UNSUPPORTED
    case .operationFailed:
        return fallback
    }
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
    timeoutSeconds: Double,
    discard: @escaping (T) -> Void = { _ in },
    work: @escaping () async throws -> T
) -> Result<T, Error> {
    let semaphore = DispatchSemaphore(value: 0)
    let result = CMAsyncResult<T>(discard: discard)

    let task = Task {
        do {
            result.store(.success(try await work()))
        } catch {
            result.store(.failure(error))
        }
        semaphore.signal()
    }

    if timeoutSeconds > 0 {
        let deadline = DispatchTime.now() + min(timeoutSeconds, 1e9)
        if semaphore.wait(timeout: deadline) == .timedOut {
            task.cancel()
            result.abandon()
            return .failure(
                CMBridgeError.timedOut("timed out waiting for CoreML async work; the work was cancelled"))
        }
    } else {
        semaphore.wait()
    }

    return result.take()
        ?? .failure(CMBridgeError.operationFailed("CoreML async work produced no result"))
}

func cm_json_object(from jsonPtr: UnsafePointer<CChar>?) throws -> [String: Any] {
    guard let jsonPtr else { return [:] }
    let json = String(cString: jsonPtr)
    guard !json.isEmpty else { return [:] }
    guard let data = json.data(using: .utf8) else {
        throw CMBridgeError.invalidArgument("failed to decode UTF-8 JSON payload")
    }
    let object = try JSONSerialization.jsonObject(with: data, options: [])
    guard let dictionary = object as? [String: Any] else {
        throw CMBridgeError.invalidArgument("expected a JSON object payload")
    }
    return dictionary
}

func cm_json_array(from jsonPtr: UnsafePointer<CChar>?) throws -> [Any] {
    guard let jsonPtr else { return [] }
    let json = String(cString: jsonPtr)
    guard !json.isEmpty else { return [] }
    guard let data = json.data(using: .utf8) else {
        throw CMBridgeError.invalidArgument("failed to decode UTF-8 JSON payload")
    }
    let object = try JSONSerialization.jsonObject(with: data, options: [])
    guard let array = object as? [Any] else {
        throw CMBridgeError.invalidArgument("expected a JSON array payload")
    }
    return array
}

func cm_json_safe(_ value: Any) -> Any {
    switch value {
    case let dictionary as [String: Any]:
        return dictionary.mapValues(cm_json_safe)
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

func cm_json_string_if_valid(_ value: Any) -> String? {
    guard JSONSerialization.isValidJSONObject(value),
        let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    else {
        return nil
    }
    return String(data: data, encoding: .utf8)
}

func cm_json_string(_ value: Any) -> String {
    cm_json_string_if_valid(value) ?? "{}"
}

func cm_multi_array_data_type(from rawValue: Int) throws -> MLMultiArrayDataType {
    switch rawValue {
    case 0x10000 | 64:
        return .float64
    case 0x10000 | 32:
        return .float32
    case 0x10000 | 16:
        return .float16
    case 0x20000 | 32:
        return .int32
    case 0x20000 | 8:
        if #available(macOS 26.0, *), let dataType = MLMultiArrayDataType(rawValue: rawValue) {
            return dataType
        }
        throw CMBridgeError.unsupported("Int8 MLMultiArray values require macOS 26.0+")
    default:
        throw CMBridgeError.invalidArgument("unsupported MLMultiArray data type: \(rawValue)")
    }
}

func cm_multi_array_data_type_object(_ dataType: MLMultiArrayDataType) -> Any {
    switch dataType.rawValue {
    case 0x10000 | 64:
        return "float64"
    case 0x10000 | 32:
        return "float32"
    case 0x10000 | 16:
        return "float16"
    case 0x20000 | 32:
        return "int32"
    case 0x20000 | 8:
        return "int8"
    default:
        return ["unknown": dataType.rawValue]
    }
}

func cm_feature_type(from rawValue: Int32) -> MLFeatureType? {
    MLFeatureType(rawValue: Int(rawValue))
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

func cm_parameter_key(named name: String) -> MLParameterKey? {
    switch name {
    case "learning_rate":
        return .learningRate
    case "momentum":
        return .momentum
    case "mini_batch_size":
        return .miniBatchSize
    case "beta1":
        return .beta1
    case "beta2":
        return .beta2
    case "eps":
        return .eps
    case "epochs":
        return .epochs
    case "shuffle":
        return .shuffle
    case "seed":
        return .seed
    case "number_of_neighbors":
        return .numberOfNeighbors
    case "linked_model_file_name":
        return .linkedModelFileName
    case "linked_model_search_path":
        return .linkedModelSearchPath
    case "weights":
        return .weights
    case "biases":
        return .biases
    default:
        let parts = name.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: true)
        if parts.count == 2, let base = cm_parameter_key(named: String(parts[0])) {
            return base.scoped(to: String(parts[1]))
        }
        return nil
    }
}

func cm_task_state_name(_ state: MLTaskState) -> String {
    switch state {
    case .suspended:
        return "suspended"
    case .running:
        return "running"
    case .cancelling:
        return "cancelling"
    case .completed:
        return "completed"
    case .failed:
        return "failed"
    @unknown default:
        return "unknown"
    }
}
