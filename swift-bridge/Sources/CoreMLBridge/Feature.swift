import CoreML
import Foundation

func cm_image_crop_and_scale_raw_value(_ name: String) -> Int? {
  switch name {
  case "center_crop":
    return 0
  case "scale_fit":
    return 1
  case "scale_fill":
    return 2
  case "scale_fit_rotate90_ccw":
    return 0x100 + 1
  case "scale_fill_rotate90_ccw":
    return 0x100 + 2
  default:
    return nil
  }
}

func cm_double(_ value: Any?) -> Double? {
  switch value {
  case let number as NSNumber:
    return number.doubleValue
  case let value as Double:
    return value
  case let value as Float:
    return Double(value)
  case let value as Int:
    return Double(value)
  default:
    return nil
  }
}

func cm_image_feature_options(
  from jsonPtr: UnsafePointer<CChar>?
) throws -> [MLFeatureValue.ImageOption: Any]? {
  let object = try cm_json_object(from: jsonPtr)
  guard !object.isEmpty else { return nil }

  var options: [MLFeatureValue.ImageOption: Any] = [:]

  if let cropRectObject = object["crop_rect"] as? [String: Any] {
    guard
      let x = cm_double(cropRectObject["x"]),
      let y = cm_double(cropRectObject["y"]),
      let width = cm_double(cropRectObject["width"]),
      let height = cm_double(cropRectObject["height"])
    else {
      throw CMBridgeError.invalidArgument("crop_rect must contain numeric x/y/width/height fields")
    }
    options[.cropRect] = CGRect(x: x, y: y, width: width, height: height)
  }

  if let cropAndScale = object["crop_and_scale"] as? String {
    guard let rawValue = cm_image_crop_and_scale_raw_value(cropAndScale) else {
      throw CMBridgeError.invalidArgument("unsupported crop_and_scale option: \(cropAndScale)")
    }
    options[.cropAndScale] = rawValue
  }

  return options.isEmpty ? nil : options
}

@_cdecl("cm_feature_new_int64")
public func cm_feature_new_int64(_ value: Int64) -> UnsafeMutableRawPointer? {
  cm_retain(MLFeatureValue(int64: value))
}

@_cdecl("cm_feature_new_double")
public func cm_feature_new_double(_ value: Double) -> UnsafeMutableRawPointer? {
  cm_retain(MLFeatureValue(double: value))
}

@_cdecl("cm_feature_new_string")
public func cm_feature_new_string(_ valuePtr: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
  guard let valuePtr else { return nil }
  return cm_retain(MLFeatureValue(string: String(cString: valuePtr)))
}

@_cdecl("cm_feature_new_multi_array")
public func cm_feature_new_multi_array(_ arrayPtr: UnsafeMutableRawPointer?)
  -> UnsafeMutableRawPointer?
{
  guard let arrayPtr else { return nil }
  let array: MLMultiArray = cm_borrow(arrayPtr)
  return cm_retain(MLFeatureValue(multiArray: array))
}

@_cdecl("cm_feature_new_image_at_url")
public func cm_feature_new_image_at_url(
  _ pathPtr: UnsafePointer<CChar>?,
  _ pixelsWide: Int,
  _ pixelsHigh: Int,
  _ pixelFormatType: UInt32,
  _ optionsJson: UnsafePointer<CChar>?,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
  guard let pathPtr else {
    cm_write_error(errorOut, "image path must not be null")
    return nil
  }
  do {
    let options = try cm_image_feature_options(from: optionsJson)
    return cm_retain(
      try MLFeatureValue(
        imageAt: cm_url(from: pathPtr),
        pixelsWide: pixelsWide,
        pixelsHigh: pixelsHigh,
        pixelFormatType: pixelFormatType,
        options: options
      )
    )
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return nil
  }
}

@_cdecl("cm_feature_new_sequence")
public func cm_feature_new_sequence(_ sequencePtr: UnsafeMutableRawPointer?)
  -> UnsafeMutableRawPointer?
{
  guard let sequencePtr else { return nil }
  let sequence: MLSequence = cm_borrow(sequencePtr)
  return cm_retain(MLFeatureValue(sequence: sequence))
}

@_cdecl("cm_feature_new_undefined")
public func cm_feature_new_undefined(_ featureTypeRaw: Int32) -> UnsafeMutableRawPointer? {
  guard let featureType = cm_feature_type(from: featureTypeRaw) else { return nil }
  return cm_retain(MLFeatureValue(undefined: featureType))
}

@_cdecl("cm_feature_new_string_dictionary")
public func cm_feature_new_string_dictionary(
  _ keys: UnsafePointer<UnsafePointer<CChar>?>?,
  _ values: UnsafePointer<Double>?,
  _ len: Int,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
  guard let keys, let values else {
    cm_write_error(errorOut, "dictionary keys and values must not be null")
    return nil
  }

  var dictionary: [String: NSNumber] = [:]
  dictionary.reserveCapacity(len)
  for index in 0..<len {
    guard let keyPtr = keys[index] else {
      cm_write_error(errorOut, "dictionary key at index \(index) was null")
      return nil
    }
    dictionary[String(cString: keyPtr)] = NSNumber(value: values[index])
  }

  do {
    return cm_retain(try MLFeatureValue(dictionary: dictionary))
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return nil
  }
}

@_cdecl("cm_feature_new_int64_dictionary")
public func cm_feature_new_int64_dictionary(
  _ keys: UnsafePointer<Int64>?,
  _ values: UnsafePointer<Double>?,
  _ len: Int,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
  guard let keys, let values else {
    cm_write_error(errorOut, "dictionary keys and values must not be null")
    return nil
  }

  var dictionary: [NSNumber: NSNumber] = [:]
  dictionary.reserveCapacity(len)
  for index in 0..<len {
    dictionary[NSNumber(value: keys[index])] = NSNumber(value: values[index])
  }

  do {
    return cm_retain(try MLFeatureValue(dictionary: dictionary))
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return nil
  }
}

@_cdecl("cm_feature_type")
public func cm_feature_type(_ featurePtr: UnsafeMutableRawPointer?) -> Int32 {
  guard let featurePtr else { return Int32(MLFeatureType.invalid.rawValue) }
  let feature: MLFeatureValue = cm_borrow(featurePtr)
  return Int32(feature.type.rawValue)
}

@_cdecl("cm_feature_is_undefined")
public func cm_feature_is_undefined(_ featurePtr: UnsafeMutableRawPointer?) -> Bool {
  guard let featurePtr else { return false }
  let feature: MLFeatureValue = cm_borrow(featurePtr)
  return feature.isUndefined
}

@_cdecl("cm_feature_get_int64")
public func cm_feature_get_int64(
  _ featurePtr: UnsafeMutableRawPointer?,
  _ outValue: UnsafeMutablePointer<Int64>?
) -> Bool {
  guard let featurePtr, let outValue else { return false }
  let feature: MLFeatureValue = cm_borrow(featurePtr)
  guard feature.type == .int64 else { return false }
  outValue.pointee = feature.int64Value
  return true
}

@_cdecl("cm_feature_get_double")
public func cm_feature_get_double(
  _ featurePtr: UnsafeMutableRawPointer?,
  _ outValue: UnsafeMutablePointer<Double>?
) -> Bool {
  guard let featurePtr, let outValue else { return false }
  let feature: MLFeatureValue = cm_borrow(featurePtr)
  guard feature.type == .double else { return false }
  outValue.pointee = feature.doubleValue
  return true
}

@_cdecl("cm_feature_get_string")
public func cm_feature_get_string(_ featurePtr: UnsafeMutableRawPointer?) -> UnsafeMutablePointer<
  CChar
>? {
  guard let featurePtr else { return nil }
  let feature: MLFeatureValue = cm_borrow(featurePtr)
  guard feature.type == .string else { return nil }
  return cm_string(feature.stringValue)
}

@_cdecl("cm_feature_get_multi_array")
public func cm_feature_get_multi_array(_ featurePtr: UnsafeMutableRawPointer?)
  -> UnsafeMutableRawPointer?
{
  guard let featurePtr else { return nil }
  let feature: MLFeatureValue = cm_borrow(featurePtr)
  guard let value = feature.multiArrayValue else { return nil }
  return cm_retain(value)
}

@_cdecl("cm_feature_get_sequence")
public func cm_feature_get_sequence(_ featurePtr: UnsafeMutableRawPointer?)
  -> UnsafeMutableRawPointer?
{
  guard let featurePtr else { return nil }
  let feature: MLFeatureValue = cm_borrow(featurePtr)
  guard let value = feature.sequenceValue else { return nil }
  return cm_retain(value)
}

@_cdecl("cm_feature_get_string_dictionary_json")
public func cm_feature_get_string_dictionary_json(_ featurePtr: UnsafeMutableRawPointer?)
  -> UnsafeMutablePointer<CChar>?
{
  guard let featurePtr else { return nil }
  let feature: MLFeatureValue = cm_borrow(featurePtr)
  guard feature.type == .dictionary else { return nil }
  let dictionary = feature.dictionaryValue
  var stringDictionary: [String: Double] = [:]
  for (key, value) in dictionary {
    guard let key = key as? String else {
      return nil
    }
    stringDictionary[key] = value.doubleValue
  }
  return cm_string(cm_json_string(stringDictionary))
}

@_cdecl("cm_feature_get_int64_dictionary_json")
public func cm_feature_get_int64_dictionary_json(_ featurePtr: UnsafeMutableRawPointer?)
  -> UnsafeMutablePointer<CChar>?
{
  guard let featurePtr else { return nil }
  let feature: MLFeatureValue = cm_borrow(featurePtr)
  guard feature.type == .dictionary else { return nil }
  let dictionary = feature.dictionaryValue
  var intDictionary: [String: Double] = [:]
  for (key, value) in dictionary {
    guard let key = key as? NSNumber else {
      return nil
    }
    intDictionary[String(key.int64Value)] = value.doubleValue
  }
  return cm_string(cm_json_string(intDictionary))
}

@_cdecl("cm_feature_is_equal")
public func cm_feature_is_equal(
  _ lhsPtr: UnsafeMutableRawPointer?,
  _ rhsPtr: UnsafeMutableRawPointer?
) -> Bool {
  guard let lhsPtr, let rhsPtr else { return false }
  let lhs: MLFeatureValue = cm_borrow(lhsPtr)
  let rhs: MLFeatureValue = cm_borrow(rhsPtr)
  return lhs.isEqual(to: rhs)
}
