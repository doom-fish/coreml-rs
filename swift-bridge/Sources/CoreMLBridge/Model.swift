import CoreML
import Foundation

@_cdecl("cm_model_load")
public func cm_model_load(
  _ pathPtr: UnsafePointer<CChar>?,
  _ configurationJson: UnsafePointer<CChar>?,
  _ outModel: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  outModel.pointee = nil
  guard let pathPtr else {
    cm_write_error(errorOut, "model path must not be null")
    return CM_INVALID_ARGUMENT
  }

  do {
    let configuration = try cm_make_configuration(from: configurationJson)
    let model = try MLModel(contentsOf: cm_url(from: pathPtr), configuration: configuration)
    outModel.pointee = cm_retain(model)
    return CM_OK
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return cm_status_code(for: error, fallback: CM_MODEL_LOAD_FAILED)
  }
}

@_cdecl("cm_model_load_from_specification")
public func cm_model_load_from_specification(
  _ bytesPtr: UnsafePointer<UInt8>?,
  _ byteCount: Int,
  _ configurationJson: UnsafePointer<CChar>?,
  _ outModel: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  outModel.pointee = nil
  guard let bytesPtr else {
    cm_write_error(errorOut, "specification bytes must not be null")
    return CM_INVALID_ARGUMENT
  }

  do {
    let data = Data(bytes: bytesPtr, count: byteCount)
    let asset = try MLModelAsset(specification: data)
    let configuration = try cm_make_configuration(from: configurationJson)
    switch cm_block_on_async(work: {
      try await MLModel.load(asset: asset, configuration: configuration)
    }) {
    case .success(let model):
      outModel.pointee = cm_retain(model)
      return CM_OK
    case .failure(let error):
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_MODEL_ASSET_FAILED)
    }
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return cm_status_code(for: error, fallback: CM_MODEL_ASSET_FAILED)
  }
}

@_cdecl("cm_model_predict")
public func cm_model_predict(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ inputsPtr: UnsafeMutableRawPointer?,
  _ outProvider: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  cm_model_predict_with_options(modelPtr, inputsPtr, nil, outProvider, errorOut)
}

@_cdecl("cm_model_predict_with_options")
public func cm_model_predict_with_options(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ inputsPtr: UnsafeMutableRawPointer?,
  _ predictionOptionsJson: UnsafePointer<CChar>?,
  _ outProvider: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  outProvider.pointee = nil
  guard let modelPtr, let inputsPtr else {
    cm_write_error(errorOut, "model and inputs must not be null")
    return CM_INVALID_ARGUMENT
  }

  let model: MLModel = cm_borrow(modelPtr)
  let inputs: CMFeatureProviderBox = cm_borrow(inputsPtr)

  do {
    let options = try cm_make_prediction_options(from: predictionOptionsJson)
    let output = try model.prediction(from: inputs, options: options)
    outProvider.pointee = cm_retain(CMFeatureProviderBox(provider: output))
    return CM_OK
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return cm_status_code(for: error, fallback: CM_PREDICTION_FAILED)
  }
}

@_cdecl("cm_model_predict_batch")
public func cm_model_predict_batch(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ batchPtr: UnsafeMutableRawPointer?,
  _ outBatch: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  cm_model_predict_batch_with_options(modelPtr, batchPtr, nil, outBatch, errorOut)
}

@_cdecl("cm_model_predict_batch_with_options")
public func cm_model_predict_batch_with_options(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ batchPtr: UnsafeMutableRawPointer?,
  _ predictionOptionsJson: UnsafePointer<CChar>?,
  _ outBatch: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  outBatch.pointee = nil
  guard let modelPtr, let batchPtr else {
    cm_write_error(errorOut, "model and batch inputs must not be null")
    return CM_INVALID_ARGUMENT
  }

  let model: MLModel = cm_borrow(modelPtr)
  let batch: CMBatchProviderBox = cm_borrow(batchPtr)

  do {
    let options = try cm_make_prediction_options(from: predictionOptionsJson)
    let output = try model.predictions(from: batch, options: options)
    outBatch.pointee = cm_retain(CMBatchProviderBox(batch: output))
    return CM_OK
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return cm_status_code(for: error, fallback: CM_PREDICTION_FAILED)
  }
}

@_cdecl("cm_model_write_to_url")
public func cm_model_write_to_url(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ pathPtr: UnsafePointer<CChar>?,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  guard let modelPtr, let pathPtr else {
    cm_write_error(errorOut, "model and destination path must not be null")
    return CM_INVALID_ARGUMENT
  }

  let model: MLModel = cm_borrow(modelPtr)
  guard let writable = model as? any MLWritable else {
    cm_write_error(errorOut, "loaded model does not conform to MLWritable")
    return CM_UNSUPPORTED
  }

  do {
    try writable.write(to: cm_url(from: pathPtr))
    return CM_OK
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return cm_status_code(for: error, fallback: CM_UPDATE_FAILED)
  }
}
