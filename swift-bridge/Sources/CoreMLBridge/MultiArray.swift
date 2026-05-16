import CoreML
import Foundation

@_cdecl("cm_multi_array_new")
public func cm_multi_array_new(
    _ shapePtr: UnsafePointer<Int64>?,
    _ rank: Int,
    _ dataTypeRaw: Int32,
    _ outArray: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    outArray.pointee = nil
    guard let shapePtr else {
        cm_write_error(errorOut, "shape pointer must not be null")
        return CM_INVALID_ARGUMENT
    }
    guard let dataType = cm_multi_array_data_type(from: dataTypeRaw) else {
        cm_write_error(errorOut, "unsupported MLMultiArray data type: \(dataTypeRaw)")
        return CM_INVALID_ARGUMENT
    }

    let shape = (0..<rank).map { NSNumber(value: shapePtr[$0]) }

    do {
        let array = try MLMultiArray(shape: shape, dataType: dataType)
        outArray.pointee = cm_retain(array)
        return CM_OK
    } catch {
        cm_write_error(errorOut, error.localizedDescription)
        return CM_MULTI_ARRAY_FAILED
    }
}

@_cdecl("cm_multi_array_count")
public func cm_multi_array_count(_ arrayPtr: UnsafeMutableRawPointer?) -> Int {
    guard let arrayPtr else { return 0 }
    let array: MLMultiArray = cm_borrow(arrayPtr)
    return array.count
}

@_cdecl("cm_multi_array_data_type")
public func cm_multi_array_data_type(_ arrayPtr: UnsafeMutableRawPointer?) -> Int32 {
    guard let arrayPtr else { return 0 }
    let array: MLMultiArray = cm_borrow(arrayPtr)
    return Int32(array.dataType.rawValue)
}

@_cdecl("cm_multi_array_rank")
public func cm_multi_array_rank(_ arrayPtr: UnsafeMutableRawPointer?) -> Int {
    guard let arrayPtr else { return 0 }
    let array: MLMultiArray = cm_borrow(arrayPtr)
    return array.shape.count
}

@_cdecl("cm_multi_array_copy_shape")
public func cm_multi_array_copy_shape(
    _ arrayPtr: UnsafeMutableRawPointer?,
    _ outBuffer: UnsafeMutablePointer<Int64>?,
    _ capacity: Int
) -> Int {
    guard let arrayPtr, let outBuffer else { return 0 }
    let array: MLMultiArray = cm_borrow(arrayPtr)
    let count = min(capacity, array.shape.count)
    for index in 0..<count {
        outBuffer[index] = array.shape[index].int64Value
    }
    return count
}

@_cdecl("cm_multi_array_copy_strides")
public func cm_multi_array_copy_strides(
    _ arrayPtr: UnsafeMutableRawPointer?,
    _ outBuffer: UnsafeMutablePointer<Int64>?,
    _ capacity: Int
) -> Int {
    guard let arrayPtr, let outBuffer else { return 0 }
    let array: MLMultiArray = cm_borrow(arrayPtr)
    let count = min(capacity, array.strides.count)
    for index in 0..<count {
        outBuffer[index] = array.strides[index].int64Value
    }
    return count
}

@_cdecl("cm_multi_array_data_pointer")
public func cm_multi_array_data_pointer(_ arrayPtr: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let arrayPtr else { return nil }
    let array: MLMultiArray = cm_borrow(arrayPtr)
    return array.dataPointer
}
