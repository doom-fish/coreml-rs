import CoreML
import CoreVideo
import Foundation

final class CMFeatureProviderBox: NSObject, MLFeatureProvider {
    fileprivate var values: [String: MLFeatureValue]

    init(values: [String: MLFeatureValue] = [:]) {
        self.values = values
    }

    convenience init(provider: any MLFeatureProvider) {
        var values: [String: MLFeatureValue] = [:]
        for name in provider.featureNames {
            if let value = provider.featureValue(for: name) {
                values[name] = value
            }
        }
        self.init(values: values)
    }

    var featureNames: Set<String> {
        Set(values.keys)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        values[featureName]
    }
}

@_cdecl("cm_feature_provider_new")
public func cm_feature_provider_new() -> UnsafeMutableRawPointer? {
    cm_retain(CMFeatureProviderBox())
}

@_cdecl("cm_feature_provider_insert_feature")
public func cm_feature_provider_insert_feature(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?,
    _ featurePtr: UnsafeMutableRawPointer?
) -> Int32 {
    guard let providerPtr, let namePtr, let featurePtr else {
        return CM_INVALID_ARGUMENT
    }

    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    let feature: MLFeatureValue = cm_borrow(featurePtr)
    provider.values[String(cString: namePtr)] = feature
    return CM_OK
}

@_cdecl("cm_feature_provider_insert_multi_array")
public func cm_feature_provider_insert_multi_array(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?,
    _ arrayPtr: UnsafeMutableRawPointer?
) -> Int32 {
    guard let providerPtr, let namePtr, let arrayPtr else {
        return CM_INVALID_ARGUMENT
    }

    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    let array: MLMultiArray = cm_borrow(arrayPtr)
    provider.values[String(cString: namePtr)] = MLFeatureValue(multiArray: array)
    return CM_OK
}

@_cdecl("cm_feature_provider_insert_pixel_buffer")
public func cm_feature_provider_insert_pixel_buffer(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?,
    _ pixelBufferPtr: UnsafeMutableRawPointer?
) -> Int32 {
    guard let providerPtr, let namePtr, let pixelBufferPtr else {
        return CM_INVALID_ARGUMENT
    }

    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    let pixelBuffer = unsafeBitCast(pixelBufferPtr, to: CVPixelBuffer.self)
    provider.values[String(cString: namePtr)] = MLFeatureValue(pixelBuffer: pixelBuffer)
    return CM_OK
}

@_cdecl("cm_feature_provider_insert_string")
public func cm_feature_provider_insert_string(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?,
    _ valuePtr: UnsafePointer<CChar>?
) -> Int32 {
    guard let providerPtr, let namePtr, let valuePtr else {
        return CM_INVALID_ARGUMENT
    }

    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    provider.values[String(cString: namePtr)] = MLFeatureValue(string: String(cString: valuePtr))
    return CM_OK
}

@_cdecl("cm_feature_provider_insert_int64")
public func cm_feature_provider_insert_int64(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?,
    _ value: Int64
) -> Int32 {
    guard let providerPtr, let namePtr else {
        return CM_INVALID_ARGUMENT
    }

    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    provider.values[String(cString: namePtr)] = MLFeatureValue(int64: value)
    return CM_OK
}

@_cdecl("cm_feature_provider_insert_double")
public func cm_feature_provider_insert_double(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?,
    _ value: Double
) -> Int32 {
    guard let providerPtr, let namePtr else {
        return CM_INVALID_ARGUMENT
    }

    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    provider.values[String(cString: namePtr)] = MLFeatureValue(double: value)
    return CM_OK
}

@_cdecl("cm_feature_provider_keys_json")
public func cm_feature_provider_keys_json(_ providerPtr: UnsafeMutableRawPointer?) -> UnsafeMutablePointer<CChar>? {
    guard let providerPtr else {
        return cm_string("[]")
    }
    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    return cm_string(cm_json_string(Array(provider.featureNames).sorted()))
}

@_cdecl("cm_feature_provider_feature_type")
public func cm_feature_provider_feature_type(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?
) -> Int32 {
    guard let providerPtr, let namePtr else {
        return -1
    }
    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    guard let value = provider.featureValue(for: String(cString: namePtr)) else {
        return -1
    }
    return Int32(value.type.rawValue)
}

@_cdecl("cm_feature_provider_get_feature")
public func cm_feature_provider_get_feature(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let providerPtr, let namePtr else {
        return nil
    }
    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    guard let value = provider.featureValue(for: String(cString: namePtr)) else {
        return nil
    }
    return cm_retain(value)
}

@_cdecl("cm_feature_provider_get_multi_array")
public func cm_feature_provider_get_multi_array(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let providerPtr, let namePtr else {
        return nil
    }
    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    guard let value = provider.featureValue(for: String(cString: namePtr))?.multiArrayValue else {
        return nil
    }
    return cm_retain(value)
}

@_cdecl("cm_feature_provider_get_string")
public func cm_feature_provider_get_string(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?
) -> UnsafeMutablePointer<CChar>? {
    guard let providerPtr, let namePtr else {
        return nil
    }
    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    guard let value = provider.featureValue(for: String(cString: namePtr)), value.type == .string else {
        return nil
    }
    return cm_string(value.stringValue)
}

@_cdecl("cm_feature_provider_get_int64")
public func cm_feature_provider_get_int64(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?,
    _ outValue: UnsafeMutablePointer<Int64>?
) -> Bool {
    guard let providerPtr, let namePtr, let outValue else {
        return false
    }
    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    guard let value = provider.featureValue(for: String(cString: namePtr)), value.type == .int64 else {
        return false
    }
    outValue.pointee = value.int64Value
    return true
}

@_cdecl("cm_feature_provider_get_double")
public func cm_feature_provider_get_double(
    _ providerPtr: UnsafeMutableRawPointer?,
    _ namePtr: UnsafePointer<CChar>?,
    _ outValue: UnsafeMutablePointer<Double>?
) -> Bool {
    guard let providerPtr, let namePtr, let outValue else {
        return false
    }
    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    guard let value = provider.featureValue(for: String(cString: namePtr)), value.type == .double else {
        return false
    }
    outValue.pointee = value.doubleValue
    return true
}
