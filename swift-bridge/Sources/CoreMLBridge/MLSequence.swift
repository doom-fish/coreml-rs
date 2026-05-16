import CoreML
import Foundation

@_cdecl("cm_sequence_new_empty")
public func cm_sequence_new_empty(
  _ featureTypeRaw: Int32,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
  guard let featureType = cm_feature_type(from: featureTypeRaw) else {
    cm_write_error(errorOut, "unsupported MLFeatureType raw value: \(featureTypeRaw)")
    return nil
  }
  switch featureType {
  case .string, .int64:
    return cm_retain(MLSequence(empty: featureType))
  default:
    cm_write_error(errorOut, "MLSequence supports only string and int64 feature types")
    return nil
  }
}

@_cdecl("cm_sequence_new_strings")
public func cm_sequence_new_strings(
  _ valuePtrs: UnsafePointer<UnsafePointer<CChar>?>?,
  _ len: Int,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
  guard let valuePtrs else {
    cm_write_error(errorOut, "sequence values must not be null")
    return nil
  }
  var values: [String] = []
  values.reserveCapacity(len)
  for index in 0..<len {
    guard let valuePtr = valuePtrs[index] else {
      cm_write_error(errorOut, "sequence string at index \(index) was null")
      return nil
    }
    values.append(String(cString: valuePtr))
  }
  return cm_retain(MLSequence(strings: values))
}

@_cdecl("cm_sequence_new_int64s")
public func cm_sequence_new_int64s(
  _ values: UnsafePointer<Int64>?,
  _ len: Int,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
  guard let values else {
    cm_write_error(errorOut, "sequence values must not be null")
    return nil
  }
  let buffer = Array(UnsafeBufferPointer(start: values, count: len)).map(NSNumber.init(value:))
  return cm_retain(MLSequence(int64s: buffer))
}

@_cdecl("cm_sequence_type")
public func cm_sequence_type(_ sequencePtr: UnsafeMutableRawPointer?) -> Int32 {
  guard let sequencePtr else { return Int32(MLFeatureType.invalid.rawValue) }
  let sequence: MLSequence = cm_borrow(sequencePtr)
  return Int32(sequence.type.rawValue)
}

@_cdecl("cm_sequence_get_strings_json")
public func cm_sequence_get_strings_json(_ sequencePtr: UnsafeMutableRawPointer?)
  -> UnsafeMutablePointer<CChar>?
{
  guard let sequencePtr else { return nil }
  let sequence: MLSequence = cm_borrow(sequencePtr)
  guard sequence.type == .string else { return nil }
  return cm_string(cm_json_string(sequence.stringValues))
}

@_cdecl("cm_sequence_get_int64s_json")
public func cm_sequence_get_int64s_json(_ sequencePtr: UnsafeMutableRawPointer?)
  -> UnsafeMutablePointer<CChar>?
{
  guard let sequencePtr else { return nil }
  let sequence: MLSequence = cm_borrow(sequencePtr)
  guard sequence.type == .int64 else { return nil }
  return cm_string(cm_json_string(sequence.int64Values.map(\.int64Value)))
}
